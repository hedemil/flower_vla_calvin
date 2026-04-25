import torch
from safetensors.torch import save_file
import gc
import os

# --- CONFIGURATION ---
input_ckpt = "checkpoints/libero_spatial/meanflower/original_model.ckpt"
output_safetensors = "checkpoints/libero_spatial/meanflower/model.safetensors"

def convert_vla_checkpoint(ckpt_path, save_path):
    print(f"🚀 Starting conversion: {ckpt_path}")
    
    # 1. Memory-efficient load (mmap=True prevents RAM spikes)
    checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True)
    
    # 2. Extract state_dict and immediately free the 15GB container
    raw_sd = checkpoint["state_dict"]
    del checkpoint
    gc.collect()
    print("✅ Loaded raw state dict. Cleaning keys...")

    # 3. Define the final 8-key mapping for the "outliers"
    # These are the ones that didn't follow the standard vlm. prefix pattern
    outlier_mapping = {
        'image_pos_embed.column_embeddings.weight': 'vlm.image_pos_embed.column_embeddings.weight',
        'visual_temporal_embed.pos_idx_to_embed': 'vlm.visual_temporal_embed.pos_idx_to_embed',
        'image_pos_embed.row_embeddings.weight': 'vlm.image_pos_embed.row_embeddings.weight',
        'language_model.model.shared.weight': 'vlm.language_shared.weight',
        'image_proj_norm.bias': 'vlm.image_proj_norm.bias',
        'image_projection': 'vlm.image_projection',
        'language_model.final_logits_bias': 'vlm.language_final_logits_bias',
        'image_proj_norm.weight': 'vlm.image_proj_norm.weight'
    }

    final_sd = {}
    
    for k, v in raw_sd.items():
        # A. Strip the training wrapper prefix 'vlm.' if it exists to normalize
        # We do this because your model had double nested names like vlm.vision_tower...
        clean_key = k.replace("vlm.", "") if k.startswith("vlm.") else k
        
        # B. Apply the Paper's structural mapping
        if clean_key.startswith("language_model.model.encoder"):
            # Map language backbone
            final_key = clean_key.replace("language_model.model.encoder", "vlm.language_encoder")
        elif clean_key.startswith("vision_tower"):
            # Map vision backbone
            final_key = clean_key.replace("vision_tower", "vlm.vision_tower")
        elif clean_key in outlier_mapping:
            # Map the 8 specific outliers
            final_key = outlier_mapping[clean_key]
        else:
            # Keep action_decoders and action_encoders as they are
            final_key = clean_key

        # C. Detach and Clone to break shared memory (prevents Safetensors RuntimeError)
        final_sd[final_key] = v.detach().clone().contiguous()

    # 4. Cleanup raw dict before saving to save RAM
    del raw_sd
    gc.collect()

    # 5. Save as Safetensors
    print(f"💾 Saving to {save_path}...")
    save_file(final_sd, save_path)
    print(f"✨ DONE! Final key count: {len(final_sd)}")
    
    # Show file size comparison
    orig_size = os.path.getsize(ckpt_path) / (1024**3)
    new_size = os.path.getsize(save_path) / (1024**3)
    print(f"📊 Size Reduction: {orig_size:.2f}GB -> {new_size:.2f}GB")

if __name__ == "__main__":
    convert_vla_checkpoint(input_ckpt, output_safetensors)