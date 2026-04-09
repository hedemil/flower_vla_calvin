from safetensors.torch import load_file

sd = load_file(
    "/leonardo_scratch/fast/AIFAC_P01_047/project/output/checkpoints/runs/2026-04-01/04-23-29/checkpoint_290000/model.safetensors",
    device="cpu",
)
prefixes = set(k.split(".")[0] for k in sd.keys())
print("Top-level prefixes:", prefixes)
print()
for p in sorted(prefixes):
    keys = [k for k in sd.keys() if k.startswith(p)]
    print(f"{p}: {len(keys)} keys")
    for k in keys[:5]:
        print(f"  {k}: {sd[k].shape}")
    print()
