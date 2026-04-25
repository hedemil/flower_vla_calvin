import torch
from safetensors.torch import load_file as load_safetensors

def get_ckpt_info(path, is_safetensors=False):
    if is_safetensors:
        ckpt = load_safetensors(path)
        top_keys = list(ckpt.keys())
        state_dict = ckpt
        return {'type': 'safetensors', 'top_keys': top_keys, 'state_dict': state_dict}
    else:
        ckpt = torch.load(path, map_location='cpu')
        top_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
        else:
            state_dict = {}
        return {'type': 'torch', 'top_keys': top_keys, 'state_dict': state_dict}

def compare_keys(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    only1 = keys1 - keys2
    only2 = keys2 - keys1
    return only1, only2

ckpt1_path = 'checkpoints/libero_spatial/meanflower/model.safetensors'
ckpt2_path = 'checkpoints/libero_spatial_paper/model.safetensors'

info1 = get_ckpt_info(ckpt1_path, is_safetensors=True)
info2 = get_ckpt_info(ckpt2_path, is_safetensors=True)

print(f"\n--- Checkpoint 1: {ckpt1_path} ---")
print(f"Type: {info1['type']}")
print(f"Top-level keys: {info1['top_keys'][:20]}")
print(f"State dict keys count: {len(info1['state_dict'])}")
print(f"First 20 state dict keys:")
for i, k in enumerate(list(info1['state_dict'].keys())[:20]):
    print(f"  [{i}] {k}\n")

print(f"\n--- Checkpoint 2: {ckpt2_path} ---")
print(f"Type: {info2['type']}")
print(f"Top-level keys: {info2['top_keys'][:20]}")
print(f"State dict keys count: {len(info2['state_dict'])}")
print(f"First 20 state dict keys:")
for i, k in enumerate(list(info2['state_dict'].keys())[:20]):
    print(f"  [{i}] {k}\n")

# Compare state dict keys
only1, only2 = compare_keys(info1['state_dict'], info2['state_dict'])
print("\n--- Key Comparison ---")
print(f"Keys only in Checkpoint 1 (model_final.safetensors): {list(only1)[:20]}")
print(f"Keys only in Checkpoint 2 (model.safetensors): {list(only2)[:20]}")