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
    VelocityDecoder,
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
        norm_eps: float = 1e-2,
        norm_p: float = 0.5,
        # iMF Configuration
        use_imf: bool = False,
        imf_v_weight: float = 1.0,
        imf_head_depth: int = 8,
        # Optimizer Configuration
        optimizer_type: str = "adamw",
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
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
        self._setup_vlm(vlm_path, freeze_vision_tower, freeze_florence)
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim
        self.action_type_adaln = action_type_adaln
        self.use_proprio = use_proprio
        self.use_imf = use_imf
        self.imf_head_depth = imf_head_depth

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
        self.norm_eps = norm_eps
        self.norm_p = norm_p
        self.imf_v_weight = imf_v_weight

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

        # Florence-2 ties encoder.embed_tokens.weight to model.shared.weight.
        # After deleting the decoder, the tie may break — populate the missing one.
        shared_key = "vlm.language_model.model.shared.weight"
        embed_key = "vlm.language_model.model.encoder.embed_tokens.weight"
        if shared_key in new_state_dict and embed_key not in new_state_dict:
            new_state_dict[embed_key] = new_state_dict[shared_key]
        elif embed_key in new_state_dict and shared_key not in new_state_dict:
            new_state_dict[shared_key] = new_state_dict[embed_key]

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
    
    # === Initialization Helpers ===

    def _init_flags(self, **kwargs):
        """Initialize model flags and configurations"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
            
        self.format_instruction = functools.partial(
                             generate_policy_prompt,
                             robot_name="Franka Panda",
                             action_space="Delta End-Effector",
                             num_arms="1",
                             prompt_style='minimal')
        
        self.use_adaln_cond = self.use_adaln_cond 
        self.use_readout_token = self.use_readout_token and self.use_adaln_cond
        self.use_proprio = self.use_proprio 
        self.use_second_view = self.use_second_view and self.second_view_key is not None
        self.use_cross_attn = self.use_cross_attn
        self.use_rope = self.use_rope and not self.use_nope
        self.use_nope = self.use_nope and not self.use_rope
        self.vlm_prompt_style = self.vlm_prompt_style
        self.return_act_chunk = False
        self.cfg_lambda = 1.0

    def _init_dimensions(self, **kwargs):
        """Initialize model dimensions."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.dit_dim % self.n_heads != 0:
            raise ValueError(f"dit_dim ({self.dit_dim}) must be divisible by n_heads ({self.n_heads})")

    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool):
        """Initialize and configure the Florence-2 VLM"""
        print(f"Loading Florence-2 from {vlm_path}")
        REVISION = "main"

        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, revision=REVISION, trust_remote_code=True, attn_implementation="eager")
        
        # Handle parameter freezing
        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        # Setup processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(vlm_path, revision=REVISION, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        # Create prompt embedding
        self.prompt_embeds = self._create_prompt_embed("<Flow>")
        
        # Remove unnecessary components
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head
        
        # Setup token dropout
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

    def _setup_dit_components_meanflow(
        self,
        dit_dim: int,
        n_heads: int,
        n_layers: int,
        action_dim: int,
        act_window_size: int,
        hidden_dim: int,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        use_cross_attn: bool,
        use_rope: bool,
        use_nope: bool,
        query_seq_len: int,
        rope_theta: float
    ) -> None:
        """
        Sets up DiT components for Mean Flow. Identical to _setup_dit_components
        except action_decoders use MeanFlowDecoder (h-conditioned) instead of nn.Linear.
        """
        # Initialize module dictionaries
        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        # Set up shared conditioning components
        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.h_embedder = TimestepEmbedder(dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(
            dit_dim,
            max_actions=len(self.action_space_index.action_spaces)
        )

        # Set up positional encoding if neither RoPE nor NoPE is used
        if not use_rope and not use_nope:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, act_window_size, dit_dim) * 0.1
            )

        # Set up DiT blocks
        block_kwargs = dict(
            dim=dit_dim,
            heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_cross_attn=use_cross_attn,
            use_rope=use_rope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta,
        )

        if self.use_imf:
            # Split into shared backbone + separate u/v heads (matching official iMF)
            shared_depth = n_layers - self.imf_head_depth
            self.shared_blocks = nn.ModuleList([
                FlowBlock(**block_kwargs) for _ in range(shared_depth)
            ])
            self.u_head_blocks = nn.ModuleList([
                FlowBlock(**block_kwargs) for _ in range(self.imf_head_depth)
            ])
            self.v_head_blocks = nn.ModuleList([
                FlowBlock(**block_kwargs) for _ in range(self.imf_head_depth)
            ])
        else:
            self.dit = nn.ModuleList([
                FlowBlock(**block_kwargs) for _ in range(n_layers)
            ])

        # Set up action-specific components
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)

            # Action encoder (same as rectified flow)
            self.action_encoders[action_name] = Mlp(
                in_features=input_dim,
                hidden_features=dit_dim,
                out_features=dit_dim,
                bias=True
            )
            # MeanFlowDecoder replaces nn.Linear
            self.action_decoders[action_name] = MeanFlowDecoder(
                dit_dim=dit_dim,
                action_dim=input_dim,
                hidden_dim=dit_dim * 2
            ).to(self.device)

            # Action-specific AdaLN
            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(
                    dit_dim,
                    global_conddim=dit_dim,
                    use_cross_attn=use_cross_attn
                )

            # Proprioceptive encoders
            if self.use_proprio:
                if action_name == 'bimanual_nav':
                    self.proprio_encoders[action_name] = Mlp(
                        input_dim,
                        dit_dim,
                        out_features=dit_dim,
                        drop=0.2
                    ).to(self.device)
                else:
                    self.proprio_encoders[action_name] = ZeroEncoder(
                        self.dit_dim,
                        device=self.device
                    )

        # Set up velocity decoder for iMF
        if self.use_imf:
            self.velocity_decoders = nn.ModuleDict()
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                input_dim = self.action_space_index.get_action_dim(action_idx)
                self.velocity_decoders[action_name] = VelocityDecoder(
                    dit_dim=dit_dim,
                    action_dim=input_dim,
                ).to(self.device)

        # Set up shared AdaLN if not using action-specific AdaLN
        if not self.action_type_adaln:
            self.adaln = SharedAdaLNController(
                dit_dim,
                global_conddim=dit_dim,
                use_cross_attn=use_cross_attn
            )

    @property
    def dit_blocks(self) -> nn.ModuleList:
        """Returns all DiT blocks as a single ModuleList for gradient clipping / iteration.
        Works for both iMF (shared + u-head + v-head) and non-iMF (single dit) modes."""
        if self.use_imf:
            all_blocks = nn.ModuleList(
                list(self.shared_blocks) + list(self.u_head_blocks) + list(self.v_head_blocks)
            )
            return all_blocks
        return self.dit

    def dit_parameters(self):
        """Yields all DiT block parameters for gradient clipping."""
        if self.use_imf:
            yield from self.shared_blocks.parameters()
            yield from self.u_head_blocks.parameters()
            yield from self.v_head_blocks.parameters()
        else:
            yield from self.dit.parameters()

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
        """Get parameter groups for optimizer with separate DiT/VLM settings."""
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        vlm_lr_scale = getattr(self.optimizer_config, 'vlm_lr_scale', 1.0)
        vlm_wd = getattr(self.optimizer_config, 'vlm_weight_decay', self.optimizer_config.transformer_weight_decay)

        dit_decay, dit_no_decay = [], []
        vlm_decay, vlm_no_decay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_vlm = name.startswith('vlm.')
            is_no_decay = any(nd in name.lower() for nd in no_decay)
            if is_vlm:
                (vlm_no_decay if is_no_decay else vlm_decay).append(param)
            else:
                (dit_no_decay if is_no_decay else dit_decay).append(param)

        return [
            {"params": dit_decay, "weight_decay": self.optimizer_config.transformer_weight_decay, "lr_scale": 1.0},
            {"params": dit_no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
            {"params": vlm_decay, "weight_decay": vlm_wd, "lr_scale": vlm_lr_scale},
            {"params": vlm_no_decay, "weight_decay": 0.0, "lr_scale": vlm_lr_scale},
        ]

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """Lightning training step"""
        # Get optimizer
        opt = self.optimizers()
        
        # Compute loss
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device) 
        total_bs = 0

        for modality_scope, dataset_batch in batch.items():
            self.modality_scope = modality_scope
            obs_features = self.encode_observations(dataset_batch)
            if self.use_imf:
                act_loss, losses_dict = self.imf_loss(obs_features, dataset_batch["actions"])
            else:
                act_loss, losses_dict = self.meanflow_loss(obs_features, dataset_batch["actions"])
            action_loss = action_loss + act_loss
            total_loss = total_loss + act_loss
            total_bs = total_bs + len(dataset_batch["actions"])

        total_loss = total_loss / len(batch)

        # Log metrics
        self._log_training_metrics(total_loss, action_loss, total_bs, losses_dict)
        if self.global_rank == 0 and batch_idx % 1000 == 0:
            if self.use_imf:
                logger.info(f"Step {self.global_step} (batch {batch_idx}) | loss={total_loss:.4f}, action_loss={action_loss:.4f}, "
                            f"loss_V={losses_dict['loss_V']:.4f}, loss_vc={losses_dict['loss_vc']:.4f}, "
                            f"dudt_norm={losses_dict['dudt_norm']:.4f}, cos_V_v={losses_dict['cos_V_v']:.4f}, "
                            f"cos_u_v={losses_dict['cos_u_v']:.4f}")
            else:
                logger.info(f"Step {self.global_step} (batch {batch_idx}) | loss={total_loss:.4f}, action_loss={action_loss:.4f}, "
                            f"raw_mse={losses_dict['raw_mse']:.4f}, v_loss={losses_dict['v_loss']:.4f}, "
                            f"dudt_norm={losses_dict['dudt_norm']:.4f}, cos_u_utgt={losses_dict['cos_u_utgt']:.4f}, "
                            f"cos_u_v={losses_dict['cos_u_v']:.4f}")

        # Optimization step
        # opt.zero_grad()
        # self.manual_backward(action_loss)
        
        # Clip gradients
         #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Step optimizer
         #opt.step()

        # Update learning rate
         #sch = self.lr_schedulers()
         #if sch is not None:
        #     sch.step()

        return action_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Lightning validation step."""
        output = {}
        with torch.no_grad():
            obs_features = self.encode_observations(batch)
            target_actions = batch[self.target_modality].to(self.device)
            noise_actions = torch.randn_like(target_actions, device=self.device)
            action_pred = self.sample_actions(noise_actions, obs_features, inference=True)
            val_loss = F.mse_loss(action_pred, target_actions)
            output["validation_loss"] = val_loss.item()
            return output

    # === Loss Functions ===
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

        # Interpolate: z_t = (1 - t) * x + t * e
        texp = t.view([b] + [1] * (actions.dim() - 1)).to(dtype=default_dtype)
        rexp = r.view([b] + [1] * (actions.dim() - 1)).to(dtype=default_dtype)

        # Sample noise
        e = torch.randn_like(actions, device=device).to(default_dtype)

        z = (1 - texp) * actions + texp * e
        v = e - actions  # target velocity

        # Define network function for JVP.
        # t and h are NOT detached — the full du/dt includes ∂u/∂t (through
        # t_embedder/adaLN) and ∂u/∂h (through MeanFlowDecoder's h_embedder).
        def u_func(z_input, t_input, r_input):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_input = t_input - r_input
                t_flat = t_input.view(-1)
                h_flat = h_input.view(-1)
                return self.dit_forward_meanflow(z_input, t_flat, h_flat, cond)

        # Tangent vectors: dz/dt = v, dt/dt = 1, dr/dt = 0
        dtdt = torch.ones_like(texp)
        drdt = torch.zeros_like(rexp)

        with torch.amp.autocast("cuda", enabled=False):
           
            u_pred, dudt = torch.func.jvp(
                u_func,
                (z, texp, rexp),
                (v, dtdt, drdt)
            )

            # u_tgt = v - h * du/dt
            h = (texp - rexp).clamp(min=0.0, max=1.0)
            u_tgt = (v - h * dudt).detach()


            # Compute loss only over valid dimensions
            diff = u_pred - u_tgt
            # diff = diff * valid_mask.to(dtype=default_dtype)
            loss_per_sample = (diff ** 2).sum(dim=(1, 2))
            raw_mse_per_sample = loss_per_sample.detach()

            # Adaptive weighting: normalizes loss to ~1.0 per sample.
            # This is critical for MeanFlow stability — without it, the
            # self-referential target u_tgt = v - h*du/dt creates a positive
            # feedback loop where large du/dt → large loss → large gradients
            # → even larger du/dt, causing divergence.
            adp_wt = (loss_per_sample.detach() + self.norm_eps) ** self.norm_p
            loss_per_sample = loss_per_sample / adp_wt

            loss = loss_per_sample.mean()

        # Monitor metrics
        with torch.no_grad():
            valid_u = u_pred
            valid_v = v
            valid_utgt = u_tgt
            v_loss = ((valid_u - valid_v) ** 2).mean()
            # Raw MSE before adaptive normalization — the real convergence signal
            raw_mse = raw_mse_per_sample.mean()
            # Track du/dt magnitude — if this vanishes, the model degenerates
            # to standard flow and single-step sampling will fail.
            dudt_norm = dudt.norm(dim=0).mean()
            # Prediction/target norms
            u_pred_norm = valid_u.norm(dim=0).mean()
            u_tgt_norm = valid_utgt.norm(dim=0).mean()
            # Cosine similarity: u_pred vs u_tgt (training alignment)
            cos_u_utgt = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_utgt.unsqueeze(0), dim=-1
            ).mean()
            # Cosine similarity: u_pred vs v (single-step convergence)
            cos_u_v = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_v.unsqueeze(0), dim=-1
            ).mean()

        # Check for NaN/Inf in outputs
        if torch.isnan(u_pred).any() or torch.isinf(u_pred).any():
            logger.warning(f"NaN/Inf detected in u_pred! "
                           f"u_pred stats: min={u_pred.min().item():.4f}, max={u_pred.max().item():.4f}, "
                           f"z stats: min={z.min().item():.4f}, max={z.max().item():.4f}, "
                           f"h stats: min={h.min().item():.4f}, max={h.max().item():.4f}")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning("NaN/Inf detected in loss! Clipping to prevent crash.")
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

        # Verify loss has gradient function
        if loss.grad_fn is None and loss.requires_grad:
            logger.warning("Loss requires_grad=True but has no grad_fn! "
                           "This indicates a gradient tracking issue.")
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
            "h_mean": h.mean().item(),
        }

        return loss, losses_dict
    
    def imf_loss(self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Improved MeanFlow (iMF) loss. Non-self-referential compound velocity target.
        Based on: https://github.com/Lyy-iiis/imeanflow

        Key differences from meanflow_loss:
        1. JVP tangent uses predicted velocity v_c (from velocity head at h=0)
        2. Loss target: V = u + h * sg(du/dt) trained against v = e - x
        3. Auxiliary v-loss trains the velocity head
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

        # Sample noise
        e = torch.randn_like(actions)

        z = (1 - texp) * actions + texp * e
        v = e - actions  # data velocity (target)

        # Step 1: Compute v_c (velocity prediction at h=0) for JVP tangent.
        # h=0 gives instantaneous velocity (not mean flow).
        # No gradients needed — v_c is only used as tangent direction.
        h_zero = torch.zeros_like(t)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_c_tangent = self._forward_v_only(z, t, h_zero, cond)

        # Step 2: JVP with has_aux=True — u_func returns (u, v_pred) where
        # v_pred is auxiliary (not differentiated). Matches official iMF pattern.
        # Single forward pass: shared blocks -> branch -> u-head + v-head.
        def u_func(z_input, t_input, r_input):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_input = t_input - r_input
                t_flat = t_input.view(-1)
                h_flat = h_input.view(-1)
                u, v_pred = self._forward_imf(z_input, t_flat, h_flat, cond)
            return u, v_pred

        dtdt = torch.ones_like(texp)
        drdt = torch.zeros_like(rexp)

        with torch.amp.autocast("cuda", enabled=False):
            u_pred, du_dt, v_pred = torch.func.jvp(
                u_func,
                (z, texp, rexp),
                (v_c_tangent, dtdt, drdt),
                has_aux=True,
            )

            # Step 3: Compound velocity V = u + h * sg(du/dt)
            h = (texp - rexp).clamp(min=0.0, max=1.0)
            V = u_pred + h * du_dt.detach()

            # Stop-gradient on target
            v_g = v.detach()

            # Step 4: Compound velocity loss (V vs v)
            diff_V = (V - v_g)
            loss_V_per_sample = (diff_V ** 2).sum(dim=(1, 2))
            adp_wt_V = (loss_V_per_sample.detach() + self.norm_eps) ** self.norm_p
            loss_V = (loss_V_per_sample / adp_wt_V).mean()

            # Step 5: Auxiliary velocity loss (v_pred from JVP primal vs v)
            diff_vc = (v_pred - v_g)
            loss_vc_per_sample = (diff_vc ** 2).sum(dim=(1, 2))
            adp_wt_vc = (loss_vc_per_sample.detach() + self.norm_eps) ** self.norm_p
            loss_vc = (loss_vc_per_sample / adp_wt_vc).mean()

            # Total loss: compound velocity + auxiliary velocity (matches iMF paper)
            loss = loss_V + loss_vc

        # Monitor metrics
        with torch.no_grad():
            valid_V = V
            valid_v = v
            valid_u = u_pred
            raw_mse_V = loss_V_per_sample.detach().mean()
            raw_mse_vc = loss_vc_per_sample.detach().mean()
            dudt_norm = du_dt.norm(dim=0).mean()
            cos_V_v = F.cosine_similarity(
                valid_V.unsqueeze(0), valid_v.unsqueeze(0), dim=-1
            ).mean()
            cos_u_v = F.cosine_similarity(
                valid_u.unsqueeze(0), valid_v.unsqueeze(0), dim=-1
            ).mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning("NaN/Inf detected in iMF loss! Clipping to prevent crash.")
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

        losses_dict = {
            "loss": loss.item() if not (torch.isnan(loss).any() or torch.isinf(loss).any()) else 1e6,
            "loss_V": loss_V.item(),
            "loss_vc": loss_vc.item(),
            "raw_mse_V": raw_mse_V.item(),
            "raw_mse_vc": raw_mse_vc.item(),
            "dudt_norm": dudt_norm.item(),
            "cos_V_v": cos_V_v.item(),
            "cos_u_v": cos_u_v.item(),
            "h_mean": h.mean().item(),
        }

        return loss, losses_dict
    
    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False) -> torch.Tensor:
        """
        Mean Flow single-step sampling: z_0 = z_1 - u(z_1, t=1, r=0)
        (h = t - r = 1)
        """
        b = z.size(0)
        device = z.device
        dtype = next(self.parameters()).dtype
        z = z.to(dtype=dtype)
        t_tensor = torch.ones(b, device=device, dtype=dtype)
        h_tensor = torch.ones(b, device=device, dtype=dtype)  # h = t - r = 1 - 0 = 1
        u = self.dit_forward_meanflow(z, t_tensor, h_tensor, cond)
        z = z - u
        return z.clamp(-1, 1)  
    
    def _dit_backbone(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor, cond_dict: dict):
        """Shared DiT backbone: encode actions, build conditioning, run blocks.
        When use_imf=True, runs only shared_blocks; otherwise runs all dit blocks.
        Returns (features, action_type, valid_dims, cond_kwargs)."""
        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape

        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(self.device)

        if self.use_proprio and cond_dict['proprio'] is not None:
            proprio = cond_dict['proprio'].to(default_dtype)
            proprio_embeds = self.encode_proprio(proprio, action_type, frequency_embeds.shape)
        else:
            proprio_embeds = torch.zeros_like(frequency_embeds)

        z, valid_dims = self.encode_actions(z, action_type)

        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding

        t_emb = stateless_norm(self.t_embedder(t)) + \
                stateless_norm(self.h_embedder(h)) + \
                stateless_norm(frequency_embeds).squeeze(1) + \
                stateless_norm(proprio_embeds).squeeze(1)

        cond = self.cond_linear(self.cond_norm(cond))

        if self.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb

        cx = z
        context = cond if self.use_cross_attn else None

        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)

        # Run shared blocks (iMF) or all blocks (non-iMF)
        blocks = self.shared_blocks if self.use_imf else self.dit
        for layer in blocks:
            cx = layer(cx, global_cond, context=context, is_causal=True, global_adaln=global_adaln)

        cond_kwargs = dict(global_cond=global_cond, context=context, global_adaln=global_adaln)
        return cx, action_type, valid_dims, cond_kwargs

    def _forward_imf(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor, cond_dict: dict):
        """Single forward pass for iMF: shared blocks -> branch -> u-head + v-head.
        Returns (u, v) from one pass, matching official iMF __call__."""
        cx, action_type, valid_dims, cond_kwargs = self._dit_backbone(z, t, h, cond_dict)

        # Branch from shared output
        cx_u = cx
        cx_v = cx

        for block in self.u_head_blocks:
            cx_u = block(cx_u, cond_kwargs['global_cond'], context=cond_kwargs['context'],
                         is_causal=True, global_adaln=cond_kwargs['global_adaln'])

        for block in self.v_head_blocks:
            cx_v = block(cx_v, cond_kwargs['global_cond'], context=cond_kwargs['context'],
                         is_causal=True, global_adaln=cond_kwargs['global_adaln'])

        u = self.decode_actions_meanflow(cx_u, h, action_type, valid_dims)
        v = self.decode_velocity(cx_v, action_type, valid_dims)

        return u, v

    def _forward_v_only(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor, cond_dict: dict):
        """Forward pass for v-head only: shared blocks -> v-head.
        Used in iMF pass 1 where only v_c_tangent is needed (u_head is skipped)."""
        cx, action_type, valid_dims, cond_kwargs = self._dit_backbone(z, t, h, cond_dict)

        for block in self.v_head_blocks:
            cx = block(cx, cond_kwargs['global_cond'], context=cond_kwargs['context'],
                       is_causal=True, global_adaln=cond_kwargs['global_adaln'])

        return self.decode_velocity(cx, action_type, valid_dims)

    def dit_forward_meanflow(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor, cond_dict: dict) -> torch.Tensor:
        """Forward pass for inference: backbone + u-head + MeanFlowDecoder.
        Only uses shared + u-head blocks (no v-head needed at inference)."""
        cx, action_type, valid_dims, cond_kwargs = self._dit_backbone(z, t, h, cond_dict)
        if self.use_imf:
            for block in self.u_head_blocks:
                cx = block(cx, cond_kwargs['global_cond'], context=cond_kwargs['context'],
                           is_causal=True, global_adaln=cond_kwargs['global_adaln'])
        return self.decode_actions_meanflow(cx, h, action_type, valid_dims)
    
    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """
        Encode proprioception based on action type.
        """
        batch_size = output_shape[0]
        default_dtype = next(self.parameters()).dtype

        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.device)

        encoded_proprio = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=default_dtype)

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded_proprio = self.proprio_encoders[action_name](proprio).squeeze(1)

        return encoded_proprio

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate action-specific AdaLN signals.
        """
        default_type = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        device = global_cond.device
        
        mod_signals = [
            torch.zeros(batch_size, self.dit_dim, device=device, dtype=default_type) 
            for _ in range(num_chunks)
        ]
        
        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond)
                for i, signal in enumerate(action_mod):
                    mod_signals[i] = signal
        
        return mod_signals
    
    def _create_prompt_embed(self, prompt_text):
        """Create embeddings for prompt tokens"""
        # Add special token if not in vocabulary
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        
        # Get token ID and create embedding
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)), 
            requires_grad=False
        )
    
        return prompt_embed.unsqueeze(0).unsqueeze(0)
    
    def encode_observations(self, batch: Dict) -> torch.Tensor:
        """Encode observations using Florence-2"""
        device = self.device
        default_type = next(self.parameters()).dtype
        
        
        embed_tensor = torch.zeros(len(batch["rgb_obs"]['rgb_static']), 1, 1)
        action_type_tensor = torch.ones(len(batch["rgb_obs"]['rgb_static']), self.act_window_size, 7)
        # Process primary image
        image_tensor = batch["rgb_obs"]['rgb_static']
        B, T, C, H, W = image_tensor.shape
        
        # Extract visual features
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_type)
        ).to(default_type)
        image_features = image_features.view(B, T * image_features.shape[1], -1)
        
        # Process second view if enabled
        if self.use_second_view:
            image2_tensor = batch["rgb_obs"]['rgb_gripper']
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_type)
            ).to(default_type)
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)
        
        # Get text embeddings
        # Get text embeddings once to reuse
        constructed_prompts = self.construct_prompts(batch)
        text_embeds = self._get_text_embeddings(constructed_prompts, device)
        
        # Add task prompt and aggregation tokens
        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(image_features.device)
        
        # Merge sequence
        merged_embeds = torch.cat([
            image_features,
            task_prompt,
            text_embeds.to(image_features.device)
        ], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)
        
        # Process through encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        # Apply dropout 
        features = self.vlm_token_dropout(features)

        # Prepare frequency and action space embeddings
        frequency_embeds = self.frequency_embedder(
            torch.ones_like(embed_tensor).to(device) * 3
        )
        
        # Get proprioception if enabled
        proprio = None
        if self.use_proprio and 'robot_obs' in batch:
            proprio = batch['robot_obs'].to(device).to(default_type)

        return {
            'features': features,
            'frequency_embeds': frequency_embeds,
            'action_space_embeds': None,
            'action_type': torch.ones_like(action_type_tensor), # actiont ype is always 1
            'proprio': proprio,
            'attention_mask': attention_mask,
        }
    
    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode actions using action-specific encoders."""
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(self.device)
        batch_size = z.shape[0]
        encoded = torch.zeros(batch_size, z.shape[1], self.dit_dim, device=self.device).to(default_dtype)
        
        # Track valid dimensions per type
        valid_dims = torch.zeros_like(z).to(default_dtype)
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded = self.action_encoders[action_name](z)
        
        return encoded, valid_dims

    def decode_actions_meanflow(
        self, z: torch.Tensor, h: torch.Tensor,
        action_type: torch.Tensor, valid_dims: torch.Tensor
    ) -> torch.Tensor:
        """
        Decodes latent representations into actions using MeanFlowDecoder.
        The decoder is conditioned on h = t - r.
        """
        default_dtype = next(self.parameters()).dtype
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(B, z.shape[1], max_action_dim, device=self.device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                decoded = self.action_decoders[action_name](z, h)
        return decoded
    
    def decode_velocity(
        self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor
    ) -> torch.Tensor:
        """
        Decodes latent representations into actions using MeanFlowDecoder.
        The decoder is conditioned on h = t - r.
        """
        default_dtype = next(self.parameters()).dtype
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(B, z.shape[1], max_action_dim, device=self.device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                decoded = self.velocity_decoders[action_name](z)
        return decoded
    
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

    def on_train_start(self):
        """Move model to device on training start."""
        self.to(self.device)
        self.vlm.to(self.device)

    def on_validation_start(self):
        self.eval()

    def on_validation_end(self):
        self.train()

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
        """Sample from logit-normal distribution. Math in float32 for stability."""
        rnd_normal = torch.randn(
            bz,
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
            bz,
            device=self.P_mean.device,
            dtype=next(self.parameters()).dtype
        )

    def sample_tr(self, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample timesteps t and r with constraint t >= r.
        `ratio` fraction of samples keep r != t (integral/mean-flow samples).
        The remaining (1 - ratio) fraction get r = t (instantaneous velocity).

        Returns:
            t: Sampled timesteps [B]
            r: Sampled timesteps [B]
        """
        dtype = next(self.parameters()).dtype

        t = self.noise_distribution()(b).to(device=self.device, dtype=dtype)
        r = self.noise_distribution()(b).to(device=self.device, dtype=dtype)

        # Ensure t >= r element-wise
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        # With probability (1 - ratio), collapse to velocity (r = t)
        prob = torch.rand(b, device=self.device)
        velocity_mask = prob < (1 - self.ratio)
        r = torch.where(velocity_mask, t, r)

        return t, r

    # === Logging ===

    def _log_training_metrics(self, total_loss, action_loss, total_bs, losses_dict=None):
        """Log training metrics."""
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True,
                sync_dist=False, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True,
                sync_dist=False, batch_size=total_bs)
        if losses_dict is not None:
            for key, value in losses_dict.items():
                self.log(f"train/{key}", value, on_step=False, on_epoch=True,
                        sync_dist=False, batch_size=total_bs)

    def _log_validation_metrics(self, pred_loss, val_total_act_loss_pp):
        """Log validation metrics."""
        self.log(
            f"val_act/{self.modality_scope}_act_loss_pp",
            pred_loss,
            sync_dist=False
        )
        try:
            n_modalities = len(self.trainer.datamodule.modalities)
        except AttributeError:
            n_modalities = 1
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / n_modalities,
            sync_dist=False
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
