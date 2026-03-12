import logging
import math
import os
from typing import Optional, Dict, Tuple, Union, List, Any
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from timm.layers.mlp import Mlp
from transformers import AutoProcessor, AutoModelForCausalLM

from flower.models.networks.meanflower_transformers import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    ZeroEncoder,
    FlowBlock,
    MeanFlowDecoder,
    stateless_norm
)
from flower.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from flower.callbacks.ema import EMA
from flower.models.utils import ActionIndex, generate_policy_prompt

logger = logging.getLogger(__name__)


class MeanFlowerVLA(pl.LightningModule):
    def __init__(
        self,
        # VLM Configuration
        vlm_path: str = "microsoft/Florence-2-large",
        freeze_florence: bool = False,
        freeze_vision_tower: bool = False,
        freeze_embeddings_only: bool = False,
        vlm_prompt_style: str = 'default',
        token_dropout: float = 0.2,
        cfg_dropout: float = 0.1,
        # Model Structure
        lowdim_obs_dim: int = 7,
        action_dim: int = 7,
        act_window_size: int = 10,
        multistep: int = 10,
        num_sampling_steps: int = 5,
        # Model flags
        use_second_view: bool = False,
        second_view_key: str = 'rgb_gripper',
        action_type_adaln: bool = True,
        use_causal_attention: bool = True,
        use_cross_attn: bool = True,
        use_adaln_cond: bool = False,
        use_readout_token: bool = False,
        use_proprio: bool = False,
        return_act_chunk: bool = False,
        # DiT Configuration
        sampling_type: str = 'ln',
        dit_dim: int = 512,
        n_heads: int = 16,
        n_layers: int = 12,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        # RoPE Configuration
        use_rope: bool = False,
        use_nope: bool = False,
        query_seq_len: int = 128,
        rope_theta: float = 32.0,
        # Mean Flow Configuration
        noise_dist: str = 'logit_normal',
        P_mean: float = -0.4,
        P_std: float = 1.0,
        ratio: float = 0.75,
        # Optimizer Configuration
        optimizer_type: str = "adamw",
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        # Decoupled MeanFlow Configuration
        encoder_depth: int = 0,           # 0 = standard MeanFlow, >0 = first N blocks use t, rest use r
        use_combined_loss: bool = False,   # True = FM loss + MF loss per batch
        freeze_encoder_blocks: bool = False,  # True = freeze first encoder_depth blocks
        # Pretrained weights
        load_pretrained: bool = False,
        pretrained_model_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.action_space_index = ActionIndex()

        # Initialize flags
        self._init_flags(
            use_second_view=use_second_view,
            use_causal_attention=use_causal_attention,
            use_cross_attn=use_cross_attn,
            use_adaln_cond=use_adaln_cond,
            use_readout_token=use_readout_token,
            use_rope=use_rope,
            use_nope=use_nope,
            vlm_prompt_style=vlm_prompt_style,
            token_dropout=token_dropout,
            action_type_adaln=action_type_adaln,
            sampling_type=sampling_type,
            use_proprio=use_proprio,
            return_act_chunk=return_act_chunk,
            second_view_key=second_view_key,
            cfg_dropout=cfg_dropout,
        )

        self.obs_modalities = []

        # Initialize dimensions
        self._init_dimensions(
            dit_dim=dit_dim,
            n_heads=n_heads,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            multistep=multistep,
            num_sampling_steps=num_sampling_steps,
        )

        self.target_modality = "actions"

        # Setup VLM
        self._setup_vlm(vlm_path, freeze_vision_tower, freeze_florence, freeze_embeddings_only)
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim

        # Setup DiT components (Mean Flow version with MeanFlowDecoder)
        self._setup_dit_components_meanflow(
            dit_dim=dit_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            action_dim=action_dim,
            act_window_size=act_window_size,
            hidden_dim=hidden_dim,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_cross_attn=use_cross_attn,
            use_rope=use_rope,
            use_nope=use_nope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta,
        )

        # Mean Flow specific parameters
        self.noise_dist = noise_dist
        self.ratio = ratio
        self.register_buffer("P_mean", torch.tensor(P_mean, dtype=torch.float32))
        self.register_buffer("P_std", torch.tensor(P_std, dtype=torch.float32))

        # Decoupled MeanFlow config
        self.encoder_depth = encoder_depth
        self.use_combined_loss = use_combined_loss
        self.freeze_encoder_blocks = freeze_encoder_blocks

        # Freeze encoder blocks if requested
        if self.freeze_encoder_blocks and self.encoder_depth > 0:
            for i, block in enumerate(self.dit[:self.encoder_depth]):
                for param in block.parameters():
                    param.requires_grad = False
            logger.info(f"Froze first {self.encoder_depth} DiT blocks (encoder)")

        # State tracking
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.modality_scope = "lang"

        # Optimizer config
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.optimizer_type = optimizer_type

        if load_pretrained and pretrained_model_path is not None:
            self._load_pretrained_weights(pretrained_model_path)

    # === Initialization Helpers ===

    def _init_flags(self, **kwargs):
        """Initialize model flags and configurations."""
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
        if self.sampling_type not in ['ln', 'pi_zero', 'loglogistic', 'uniform', 'stratified']:
            raise ValueError(f"Invalid sampling type: {self.sampling_type}")

        self.format_instruction = functools.partial(
            generate_policy_prompt,
            robot_name="Franka Panda",
            action_space="Delta End-Effector",
            num_arms="1",
            prompt_style='minimal'
        )

        self.use_readout_token = self.use_readout_token and self.use_adaln_cond
        self.use_second_view = self.use_second_view and self.second_view_key is not None
        self.use_rope = self.use_rope and not self.use_nope
        self.use_nope = self.use_nope and not self.use_rope
        self.cfg_lambda = 1.0

    def _init_dimensions(self, **kwargs):
        """Initialize model dimensions."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.dit_dim % self.n_heads != 0:
            raise ValueError(f"dit_dim ({self.dit_dim}) must be divisible by n_heads ({self.n_heads})")

    def _setup_vlm(self, vlm_path, freeze_vision_tower, freeze_florence, freeze_embeddings_only):
        """Initialize and configure the Florence-2 VLM."""
        logger.info(f"Loading Florence-2 from {vlm_path}")
        REVISION = "main"
        self.vlm = AutoModelForCausalLM.from_pretrained(
            vlm_path, revision=REVISION, trust_remote_code=True, attn_implementation="eager"
        )

        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif freeze_embeddings_only:
            embedding_layer = self.vlm.get_input_embeddings()
            for param in embedding_layer.parameters():
                param.requires_grad = False
            if hasattr(self.vlm.language_model, 'shared'):
                for param in self.vlm.language_model.shared.parameters():
                    param.requires_grad = False

        if not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(vlm_path, revision=REVISION, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.prompt_embeds = self._create_prompt_embed("<Flow>")
        del self.vlm.language_model.model.decoder, self.vlm.language_model.lm_head
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

    def _setup_dit_components_meanflow(
        self, dit_dim, n_heads, n_layers, action_dim, act_window_size, hidden_dim,
        attn_pdrop, resid_pdrop, mlp_pdrop, use_cross_attn,
        use_rope, use_nope, query_seq_len, rope_theta
    ):
        """
        Sets up DiT components for Mean Flow. Uses MeanFlowDecoder (h-conditioned)
        instead of nn.Linear for action decoders.
        """
        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)

            self.action_encoders[action_name] = Mlp(
                in_features=input_dim,
                hidden_features=dit_dim,
                out_features=dit_dim,
                bias=True
            )
            self.action_decoders[action_name] = MeanFlowDecoder(
                dit_dim=dit_dim,
                action_dim=input_dim,
                hidden_dim=dit_dim * 2
            )

            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(
                    dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn
                )

            if self.use_proprio:
                if action_name == 'bimanual_nav':
                    self.proprio_encoders[action_name] = Mlp(
                        input_dim, dit_dim, out_features=dit_dim, drop=0.2
                    )
                else:
                    self.proprio_encoders[action_name] = ZeroEncoder(self.dit_dim)

        if not self.action_type_adaln:
            self.adaln = SharedAdaLNController(
                dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn
            )

        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(
            dit_dim, max_actions=len(self.action_space_index.action_spaces)
        )

        if not use_rope and not use_nope:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, act_window_size, dit_dim) * 0.1
            )

        self.dit = nn.ModuleList([
            FlowBlock(
                dim=dit_dim,
                heads=n_heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_pdrop=mlp_pdrop,
                use_cross_attn=use_cross_attn,
                use_rope=use_rope,
                query_seq_len=query_seq_len,
                rope_theta=rope_theta
            ) for _ in range(n_layers)
        ])

    def _create_prompt_embed(self, prompt_text):
        """Create embeddings for prompt tokens."""
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False
        )
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def _load_pretrained_weights(self, pretrained_model_path: str):
        """Loads pretrained weights, handling key mismatches (e.g., different prefixes)."""
        print(f"Loading pretrained weights from {pretrained_model_path}...")
        if pretrained_model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_model_path, device=str(self.device))
            checkpoint = {"state_dict": state_dict}
        else:
            checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)

        state_dict = checkpoint.get("state_dict", checkpoint)

        if ("callbacks" in checkpoint and
                "EMA" in checkpoint["callbacks"] and
                "ema_weights" in checkpoint["callbacks"]["EMA"]):
            print("Found EMA weights in checkpoint, attempting to load them...")
            ema_weights_list = checkpoint['callbacks']['EMA']['ema_weights']
            original_state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {}
            ema_idx = 0
            for param_name, original_param in original_state_dict.items():
                if ema_idx < len(ema_weights_list):
                    ema_weight = ema_weights_list[ema_idx]
                    if ema_weight.shape == original_param.shape:
                        state_dict[param_name] = ema_weight
                        ema_idx += 1
                    else:
                        found_match = False
                        for temp_idx in range(ema_idx, min(ema_idx + 20, len(ema_weights_list))):
                            if ema_weights_list[temp_idx].shape == original_param.shape:
                                state_dict[param_name] = ema_weights_list[temp_idx]
                                ema_weights_list[temp_idx], ema_weights_list[ema_idx] = ema_weights_list[ema_idx], ema_weights_list[temp_idx]
                                ema_idx += 1
                                found_match = True
                                break
                        if not found_match:
                            print(f"Warning: No matching EMA weight found for {param_name}, using original")
                            state_dict[param_name] = original_param
                else:
                    print(f"Warning: Ran out of EMA weights at {param_name}, using original")
                    state_dict[param_name] = original_param
            print(f"Successfully matched {ema_idx} EMA weights out of {len(ema_weights_list)} total")

        # Fix key mismatches
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("agent.", "")
            if "vlm.language_encoder." in new_key:
                new_key = new_key.replace("vlm.language_encoder.", "vlm.language_model.model.encoder.")
            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        print(f"Pretrained weights loaded:")
        if missing_keys:
            print(f"  Missing keys (randomly initialized): {len(missing_keys)}")
            print(f"    {missing_keys[:30]} ...")
        if unexpected_keys:
            print(f"  Unexpected keys (ignored): {len(unexpected_keys)}")
            print(f"    {unexpected_keys[:30]} ...")
        if not missing_keys and not unexpected_keys:
            print("  All keys matched successfully!")
        return missing_keys, unexpected_keys

    # === Lightning Interface ===

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optim_groups = self._get_param_groups()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.betas
        )
        scheduler = TriStageLRScheduler(
            optimizer,
            OmegaConf.create(self.lr_scheduler_config)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def _get_param_groups(self):
        """Get parameter groups for optimizer with separate decoder group."""
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decoder_params_set = set()
        for decoder in self.action_decoders.values():
            decoder_params_set.update(p for p in decoder.parameters())

        decoder_weight_decay = self.optimizer_config.get(
            "decoder_weight_decay", self.optimizer_config.transformer_weight_decay
        )

        decay_group = []
        no_decay_group = []
        decoder_decay_group = []
        decoder_no_decay_group = []
        vlm_params = set(p for p in self.vlm.parameters())

        for name, param in self.named_parameters():
            if param.requires_grad and param not in vlm_params:
                is_no_decay = any(nd in name.lower() for nd in no_decay)
                if param in decoder_params_set:
                    if is_no_decay:
                        decoder_no_decay_group.append(param)
                    else:
                        decoder_decay_group.append(param)
                else:
                    if is_no_decay:
                        no_decay_group.append(param)
                    else:
                        decay_group.append(param)

        return [
            {"params": decay_group, "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": no_decay_group, "weight_decay": 0.0},
            {"params": decoder_decay_group, "weight_decay": decoder_weight_decay},
            {"params": decoder_no_decay_group, "weight_decay": 0.0},
        ]

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """Lightning training step."""
        total_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0

        for modality_scope, dataset_batch in batch.items():
            self.modality_scope = modality_scope
            obs_features = self.encode_observations(dataset_batch)

            if self.use_combined_loss:
                fm_loss, fm_dict = self.rf_loss(obs_features, dataset_batch["actions"])
                mf_loss, mf_dict = self.meanflow_loss(obs_features, dataset_batch["actions"])
                action_loss = fm_loss + mf_loss
                losses_dict = {**fm_dict, **mf_dict}
            else:
                action_loss, losses_dict = self.meanflow_loss(
                    obs_features, dataset_batch["actions"]
                )

            total_loss = total_loss + action_loss
            total_bs += len(dataset_batch["actions"])

        total_loss = total_loss / len(batch)
        self._log_training_metrics(total_loss, total_loss, total_bs, losses_dict)
        return total_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Lightning validation step."""
        output = {}
        with torch.no_grad():
            obs_features = self.encode_observations(batch)
            target_actions = batch[self.target_modality]
            noise_actions = torch.randn_like(target_actions, device=self.device)
            action_pred = self.sample_actions(noise_actions, obs_features, inference=True)
            val_loss = F.mse_loss(action_pred, target_actions)
            self._log_validation_metrics(val_loss, val_loss)
            output["validation_loss"] = val_loss / len(batch)
            return output

    def on_train_start(self):
        """Move model to device on training start."""
        self.to(self.device)
        self.vlm.to(self.device)

    def on_validation_start(self):
        self.eval()

    def on_validation_end(self):
        self.train()

    # === Encoding Methods (adapted for LIBERO batch format) ===

    def encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Encode observations using Florence-2, adapted for LIBERO batch format.
        Expects batch with keys: rgb_obs, lang_text, (optionally robot_obs).
        """
        device = self.device
        default_dtype = next(self.parameters()).dtype

        # Process primary image
        image_tensor = batch["rgb_obs"]["rgb_static"]
        B, T, C, H, W = image_tensor.shape
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_dtype)
        )
        image_features = image_features.view(B, T * image_features.shape[1], -1)

        # Process second view if enabled
        if self.use_second_view and self.second_view_key in batch["rgb_obs"]:
            image2_tensor = batch["rgb_obs"][self.second_view_key]
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_dtype)
            )
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)

        # Get text embeddings from raw strings
        constructed_prompts = self.construct_prompts(batch)
        text_embeds = self._get_text_embeddings(constructed_prompts, device)

        # Prompt token
        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(image_features.device)

        # Merge: [prompt, image, text]
        merged_embeds = torch.cat([
            task_prompt,
            image_features,
            text_embeds.to(image_features.device)
        ], dim=1)

        # Attention mask (all ones — padding is minimal with truncation)
        attention_mask = torch.ones(merged_embeds.shape[:2], dtype=torch.bool, device=device)

        # VLM encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state

        features = self.vlm_token_dropout(features)

        # CFG dropout on text features during training
        if self.cfg_dropout > 0 and self.training:
            prompt_length = task_prompt.shape[1]
            image_length = image_features.shape[1]
            text_length = text_embeds.shape[1]
            text_start = prompt_length + image_length
            text_end = text_start + text_length
            drop_mask = (torch.rand(B, device=device) < self.cfg_dropout).to(dtype=default_dtype).view(B, 1, 1)
            features[:, text_start:text_end, :] = features[:, text_start:text_end, :] * (1 - drop_mask)

        # For LIBERO: hardcode frequency=3 and action_type=1 (eef_delta)
        freq_input = torch.full((B,), 3.0, device=device, dtype=default_dtype)
        action_type = torch.ones(B, device=device, dtype=torch.long)  # eef_delta = 1

        # Proprioception
        proprio = None
        if self.use_proprio and 'robot_obs' in batch:
            proprio = batch['robot_obs'].to(device).to(default_dtype)

        return {
            'features': features,
            'frequency_embeds': self.frequency_embedder(freq_input),
            'action_space_embeds': self.action_space_embedder(action_type),
            'action_type': action_type,
            'proprio': proprio,
            'attention_mask': attention_mask,
        }

    def construct_prompts(self, dataset_batch):
        """Constructs prompts for Florence-2's encoder."""
        language_instruction = dataset_batch["lang_text"]
        text_prompts = []
        for instruction in language_instruction:
            if self.vlm_prompt_style == "default":
                text_prompts.append(self.format_instruction(instruction))
            elif self.vlm_prompt_style == "feature_focused":
                prompt = f"<od>{instruction}</od><grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
                text_prompts.append(prompt)
            elif self.vlm_prompt_style == "state_oriented":
                prompt = f"<od>{instruction}</od><referring_expression_segmentation>locate objects and regions for manipulation</referring_expression_segmentation>"
                text_prompts.append(prompt)
            else:
                raise ValueError(f"Unknown prompt style: {self.vlm_prompt_style}")
        return text_prompts

    def _get_text_embeddings(self, text, device):
        """Get text embeddings from raw strings."""
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        return self.vlm.get_input_embeddings()(text_inputs["input_ids"])

    # === Action Encoding/Decoding ===

    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes actions for each sample based on its action type."""
        action_type = action_type.to(self.device)
        B = z.shape[0]
        encoded = torch.zeros(B, z.shape[1], self.dit_dim, device=self.device, dtype=z.dtype)
        valid_dims = torch.zeros_like(z)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                valid_dims[mask, :, :adim] = 1
                encoded[mask] = self.action_encoders[action_name](z[mask, :, :adim])
        return encoded, valid_dims

    def decode_actions_meanflow(
        self, z: torch.Tensor, h: torch.Tensor,
        action_type: torch.Tensor, valid_dims: torch.Tensor
    ) -> torch.Tensor:
        """
        Decodes latent representations into actions using MeanFlowDecoder.
        The decoder is conditioned on h = t - r.
        """
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(B, z.shape[1], max_action_dim, device=self.device, dtype=z.dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                if mask.all():
                    pred = self.action_decoders[action_name](z, h)
                else:
                    h_masked = h[mask] if h.dim() >= 1 and h.shape[0] == B else h
                    pred = self.action_decoders[action_name](z[mask], h_masked)
                decoded[mask, :, :adim] = pred[..., :adim] * valid_dims[mask, :, :adim]
        return decoded

    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """Encodes proprioceptive data based on action type."""
        batch_size, _ = output_shape
        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=proprio.dtype)
        encoded = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=proprio.dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded[mask] = self.proprio_encoders[action_name](proprio[mask]).squeeze(1)
        return encoded

    # === Mean Flow Loss ===

    def meanflow_loss(self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the Mean Flow loss using JVP (Jacobian Vector Product).
        Based on: https://github.com/Gsunshine/py-meanflow
        """
        default_dtype = next(self.parameters()).dtype
        action_type = cond['action_type']
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(dtype=default_dtype)

        # Sample t and r with constraint t >= r
        t, r = self.sample_tr(b)

        texp = t.view([b] + [1] * (actions.dim() - 1)).to(dtype=default_dtype)
        rexp = r.view([b] + [1] * (actions.dim() - 1)).to(dtype=default_dtype)

        # Sample noise per action space
        e = torch.zeros_like(actions)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                noise_slice = torch.randn(
                    (mask.sum(), actions.size(1), adim),
                    dtype=actions.dtype, device=device
                )
                e[mask, :, :adim] = noise_slice

        z = (1 - texp) * actions + texp * e
        v = e - actions  # target velocity

        # Cast to float32 for JVP — dual tensors must have matching dtype
        z = z.float()
        v = v.float()
        texp = texp.float()
        rexp = rexp.float()

        # Define network function for JVP
        def u_func(z_input, t_input, r_input):
            t_flat = t_input.detach().view(-1)
            r_flat = r_input.detach().view(-1)
            return self.dit_forward_meanflow(z_input, t_flat, r_flat, cond)

        # Tangent vectors for JVP
        dtdt = torch.ones_like(texp)
        drdt = torch.zeros_like(rexp)

        # Monkey-patch nn.Linear and RmsNorm for JVP dtype safety
        _orig_linear_forward = nn.Linear.forward
        _orig_rmsnorm_forward = RmsNorm.forward

        def _jvp_safe_linear_forward(self, input):
            return F.linear(
                input,
                self.weight.to(input.dtype),
                self.bias.to(input.dtype) if self.bias is not None else None,
            )

        def _jvp_safe_rmsnorm_forward(self, x):
            return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)

        with torch.amp.autocast("cuda", enabled=False):
            nn.Linear.forward = _jvp_safe_linear_forward
            RmsNorm.forward = _jvp_safe_rmsnorm_forward
            try:
                u_pred, dudt = torch.func.jvp(
                    u_func,
                    (z, texp, rexp),
                    (v, dtdt, drdt)
                )
            finally:
                nn.Linear.forward = _orig_linear_forward
                RmsNorm.forward = _orig_rmsnorm_forward

            # u_tgt = v - h * du/dt
            h = (texp - rexp).clamp(min=0.0, max=1.0)
            u_tgt = (v - h * dudt).detach()

            # Build valid mask
            valid_mask = torch.zeros_like(actions, dtype=torch.bool)
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                mask = (action_type == action_idx)
                if mask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    mask_expanded = mask.view(-1, 1, 1).expand(-1, actions.size(1), adim).to(device)
                    valid_mask[mask, :, :adim] = mask_expanded[mask]

            # Compute loss only over valid dimensions
            diff = u_pred - u_tgt
            diff = diff * valid_mask.to(dtype=default_dtype)
            loss_per_sample = (diff ** 2).sum(dim=(1, 2))
            raw_mse_per_sample = loss_per_sample.detach()

            # Adaptive weighting for stability
            norm_eps = 0.001
            norm_p = 0.75
            adp_wt = (loss_per_sample.detach() + norm_eps) ** norm_p
            loss_per_sample = loss_per_sample / adp_wt

            loss = loss_per_sample.mean()

        # Monitor metrics
        with torch.no_grad():
            valid_u = u_pred[valid_mask]
            valid_v = v[valid_mask]
            valid_utgt = u_tgt[valid_mask]
            v_loss = ((valid_u - valid_v) ** 2).mean()
            raw_mse = raw_mse_per_sample.mean()
            dudt_norm = dudt[valid_mask].norm(dim=0).mean()
            u_pred_norm = valid_u.norm(dim=0).mean()
            u_tgt_norm = valid_utgt.norm(dim=0).mean()
            cos_u_utgt = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_utgt.unsqueeze(0), dim=-1
            ).mean()
            cos_u_v = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_v.unsqueeze(0), dim=-1
            ).mean()
            t_flat = t.view(-1)
            low_mask = t_flat < 0.3
            mid_mask = (t_flat >= 0.3) & (t_flat < 0.7)
            high_mask = t_flat >= 0.7
            u_v_diff = (u_pred - v) ** 2 * valid_mask.to(dtype=default_dtype)
            u_v_per_sample = u_v_diff.sum(dim=(1, 2))
            vloss_t_low = u_v_per_sample[low_mask].mean() if low_mask.any() else torch.tensor(0.0)
            vloss_t_mid = u_v_per_sample[mid_mask].mean() if mid_mask.any() else torch.tensor(0.0)
            vloss_t_high = u_v_per_sample[high_mask].mean() if high_mask.any() else torch.tensor(0.0)

        if torch.isnan(u_pred).any() or torch.isinf(u_pred).any():
            logger.warning(f"NaN/Inf detected in u_pred!")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning("NaN/Inf detected in loss! Clipping to prevent crash.")
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

        if loss.grad_fn is None and loss.requires_grad:
            logger.warning("Loss requires_grad=True but has no grad_fn!")
        elif not loss.requires_grad:
            logger.error("Loss does not require gradients! Setting requires_grad=True")
            loss.requires_grad_(True)

        losses_dict = {
            "loss": loss.item() if not (torch.isnan(loss).any() or torch.isinf(loss).any()) else 1e6,
            "raw_mse": raw_mse.item(),
            "v_loss": v_loss.item() if not (torch.isnan(v_loss).any() or torch.isinf(v_loss).any()) else 1e6,
            "dudt_norm": dudt_norm.item(),
            "u_pred_norm": u_pred_norm.item(),
            "u_tgt_norm": u_tgt_norm.item(),
            "cos_u_utgt": cos_u_utgt.item(),
            "cos_u_v": cos_u_v.item(),
            "vloss_t_low": vloss_t_low.item(),
            "vloss_t_mid": vloss_t_mid.item(),
            "vloss_t_high": vloss_t_high.item(),
            "h_mean": h.mean().item(),
        }

        return loss, losses_dict

    def rf_loss(self, cond: dict, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Standard rectified flow loss (velocity matching) for combined FM + MF training.
        When r = t, h = 0, mean velocity becomes instantaneous velocity.
        """
        default_dtype = next(self.parameters()).dtype
        action_type = cond['action_type']
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(dtype=default_dtype)

        # Sample t from logit-normal (same distribution as MF)
        t = torch.sigmoid(torch.randn((b,), device=device)).clamp(min=0.001, max=0.999)
        texp = t.view([b] + [1] * (actions.dim() - 1)).to(dtype=default_dtype)

        # Sample noise per action space
        noise = torch.zeros_like(actions)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                noise_slice = torch.randn(
                    (mask.sum(), actions.size(1), adim),
                    dtype=actions.dtype, device=device
                )
                noise[mask, :, :adim] = noise_slice

        z = (1 - texp) * actions + texp * noise

        # Forward with r=t (instantaneous velocity, h=0)
        v_pred = self.dit_forward_meanflow(z, t, r=t, cond_dict=cond)
        v_target = noise - actions

        # Build valid mask
        valid_mask = torch.zeros_like(actions, dtype=torch.bool)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                mask_expanded = mask.view(-1, 1, 1).expand(-1, actions.size(1), adim).to(device)
                valid_mask[mask, :, :adim] = mask_expanded[mask]

        diff = (v_pred - v_target) * valid_mask.to(dtype=default_dtype)
        loss = (diff ** 2).mean()

        return loss, {"rf_loss": loss.item()}

    # === Noise Distribution & Sampling for Mean Flow ===

    def noise_distribution(self):
        """Returns the noise distribution function based on config."""
        if self.noise_dist == 'logit_normal':
            return self._logit_normal_dist
        elif self.noise_dist == 'uniform':
            return self._uniform_dist
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def _logit_normal_dist(self, bz: int) -> torch.Tensor:
        """Sample from logit-normal distribution."""
        rnd_normal = torch.randn(
            bz, 1, 1, 1,
            device=self.P_mean.device,
            dtype=torch.float32
        )
        out = torch.sigmoid(
            rnd_normal * self.P_std.float() + self.P_mean.float()
        )
        return out.to(next(self.parameters()).dtype)

    def _uniform_dist(self, bz: int) -> torch.Tensor:
        """Sample from uniform distribution."""
        return torch.rand(
            bz, 1, 1, 1,
            device=self.P_mean.device,
            dtype=next(self.parameters()).dtype
        )

    def sample_tr(self, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample timesteps t and r with constraint t >= r.
        ratio fraction of samples keep r != t (integral/mean-flow samples).
        The remaining (1 - ratio) fraction get r = t (instantaneous velocity).
        """
        dtype = next(self.parameters()).dtype
        t = self.noise_distribution()(b).to(device=self.device, dtype=dtype)
        r = self.noise_distribution()(b).to(device=self.device, dtype=dtype)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        prob = torch.rand(b, 1, 1, 1, device=self.device)
        velocity_mask = prob < (1 - self.ratio)
        r = torch.where(velocity_mask, t, r)
        return t, r

    # === Sampling Methods ===

    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False) -> torch.Tensor:
        """Samples actions from the DiT model using single-step mean flow."""
        b = z.size(0)
        action_type = cond['action_type']
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                z[mask, :, adim:] = 0.0
        return self._sample_with_fixed_steps(z, cond, inference)

    def _sample_with_fixed_steps(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False) -> torch.Tensor:
        """
        Mean Flow single-step sampling: z_0 = z_1 - u(z_1, t=1, r=0)
        (h = t - r = 1)
        """
        b = z.size(0)
        device = z.device
        dtype = next(self.parameters()).dtype
        z = z.to(dtype=dtype)
        t_tensor = torch.ones(b, device=device, dtype=dtype)
        r_tensor = torch.zeros(b, device=device, dtype=dtype)
        u = self.dit_forward_meanflow(z, t_tensor, r_tensor, cond)
        z = z - u
        return z.clamp(-1, 1)

    # === DiT Forward ===

    def dit_forward_meanflow(self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor, cond_dict: dict) -> torch.Tensor:
        """
        Forward pass through the DiT blocks using MeanFlowDecoder.

        Args:
            z: Latent actions [B, T, action_dim]
            t: Current timestep [B]
            r: Target timestep [B] (h = t - r is computed internally)
            cond_dict: Conditioning dictionary
        """
        B, t_seq, d = z.shape
        working_dtype = z.dtype

        cond = self.cond_linear(self.cond_norm(cond_dict['features'].to(working_dtype)))
        freq_embeds = cond_dict['frequency_embeds'].squeeze(1).to(working_dtype)
        action_type = cond_dict['action_type'].to(self.device)
        proprio = cond_dict.get('proprio', torch.zeros_like(freq_embeds)).to(working_dtype) if self.use_proprio else torch.zeros_like(freq_embeds)
        proprio_embeds = self.encode_proprio(proprio, action_type, freq_embeds.shape).to(working_dtype)

        z, valid_dims = self.encode_actions(z, action_type)
        if not (self.use_rope or self.use_nope):
            z += self.positional_encoding

        # Apply CFG dropout on freq_embeds and proprio_embeds only
        if self.training and self.cfg_dropout > 0:
            drop_mask = (torch.rand(freq_embeds.size(0), device=freq_embeds.device) < self.cfg_dropout).to(dtype=working_dtype).unsqueeze(1)
            freq_embeds = freq_embeds * (1 - drop_mask)
            proprio_embeds = proprio_embeds * (1 - drop_mask)

        t_emb = sum(map(stateless_norm, [self.t_embedder(t), freq_embeds, proprio_embeds]))

        if self.use_adaln_cond:
            vision_cond = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond_t = vision_cond + t_emb
        else:
            global_cond_t = t_emb

        context = cond if self.use_cross_attn else None

        if self.encoder_depth > 0:
            # Decoupled MeanFlow: encoder blocks conditioned on t, decoder blocks on r
            r_emb = sum(map(stateless_norm, [self.t_embedder(r), freq_embeds, proprio_embeds]))
            if self.use_adaln_cond:
                global_cond_r = vision_cond + r_emb
            else:
                global_cond_r = r_emb

            global_adaln_t = self.adaln(global_cond_t) if not self.action_type_adaln else self.action_specific_adaln(global_cond_t, action_type)
            global_adaln_r = self.adaln(global_cond_r) if not self.action_type_adaln else self.action_specific_adaln(global_cond_r, action_type)

            # Encoder blocks (conditioned on t)
            for layer in self.dit[:self.encoder_depth]:
                z = layer(z, global_cond_t, context=context, custom_attn_mask=None,
                        custom_cross_attn_mask=cond_dict['attention_mask'], is_causal=True, global_adaln=global_adaln_t)

            # Decoder blocks (conditioned on r)
            for layer in self.dit[self.encoder_depth:]:
                z = layer(z, global_cond_r, context=context, custom_attn_mask=None,
                        custom_cross_attn_mask=cond_dict['attention_mask'], is_causal=True, global_adaln=global_adaln_r)
        else:
            # Standard MeanFlow: all blocks conditioned on t
            global_adaln = self.adaln(global_cond_t) if not self.action_type_adaln else self.action_specific_adaln(global_cond_t, action_type)

            for layer in self.dit:
                z = layer(z, global_cond_t, context=context, custom_attn_mask=None,
                        custom_cross_attn_mask=cond_dict['attention_mask'], is_causal=True, global_adaln=global_adaln)

        h = t - r  # Compute h for the MeanFlowDecoder
        return self.decode_actions_meanflow(z, h, action_type, valid_dims)

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """Computes action-specific AdaLN modulation signals."""
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        mod_signals = [torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=global_cond.dtype) for _ in range(num_chunks)]
        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond[mask])
                for i, signal in enumerate(action_mod):
                    mod_signals[i][mask] = signal
        return mod_signals

    # === Inference ===

    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """Inference forward pass for LIBERO evaluation."""
        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        batch = {
            "rgb_obs": {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper
            },
            "lang_text": [goal["lang_text"]]
        }
        features = self.encode_observations(batch)

        noise = torch.randn(
            len(features['features']),
            self.act_window_size,
            self.action_dim,
            device=features['features'].device
        )
        return self.sample_actions(noise, features, inference=True)

    @torch.no_grad()
    def step(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """Do one step of inference, handling action chunking."""
        if self.rollout_step_counter % self.multistep == 0:
            if getattr(self, 'use_bf16', False):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    self.pred_action_seq = self(obs, goal)
            else:
                self.pred_action_seq = self(obs, goal)

        if not self.return_act_chunk:
            current_action = self.pred_action_seq[0, self.rollout_step_counter]
            if len(current_action.shape) == 2:
                current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        else:
            current_action = self.pred_action_seq

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        return current_action

    def reset(self):
        """Reset model state for new rollout."""
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()

    # === Logging ===

    def _log_training_metrics(self, total_loss, action_loss, total_bs, losses_dict=None):
        """Log training metrics."""
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True,
                sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True,
                sync_dist=True, batch_size=total_bs)
        if losses_dict is not None:
            for key, value in losses_dict.items():
                self.log(f"train/{key}", value, on_step=False, on_epoch=True,
                        sync_dist=True, batch_size=total_bs)

    def _log_validation_metrics(self, pred_loss, val_total_act_loss_pp):
        """Log validation metrics."""
        self.log(
            f"val_act/{self.modality_scope}_act_loss_pp",
            pred_loss,
            sync_dist=True
        )
        try:
            n_modalities = len(self.trainer.datamodule.modalities)
        except AttributeError:
            n_modalities = 1
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / n_modalities,
            sync_dist=True
        )

    def print_model_parameters(self):
        """Print model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")
        for name, submodule in self.named_modules():
            if '.' not in name or name.count('.') <= 1:
                submodule_params = sum(p.numel() for p in submodule.parameters())
                if submodule_params > 0:
                    print(f"{name} - Total Params: {submodule_params}")
