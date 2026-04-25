# tools/

One-off utilities and debug helpers used during development. Not part of the
core training/evaluation flow — kept here for reference.

| File | Purpose |
|---|---|
| `convert_ckpt.py` | Convert a Lightning `.ckpt` checkpoint into a `.safetensors` file with the canonical `vlm.*` key layout the inference path expects. Edit the `input_ckpt` / `output_safetensors` paths at the top before running. |
| `test_ckpt.py` | Compare two `.safetensors` checkpoints — prints top-level keys, key counts, and which keys are unique to each. Useful for diffing a converted checkpoint against a known-good reference. |
| `inspect_chk.py` | Quick inspection of a `.safetensors` checkpoint — prints the second-level prefixes under `agent.*` and a sample of keys per prefix. |
| `sbatch_debug.sh` | Generic 30-minute debug job for Leonardo HPC. Validates env, GPU, HF cache, Florence-2 load, then runs 10 batches of training. `sbatch tools/sbatch_debug.sh libero meanflower`. |
| `sbatch_train_libero_imf_debug.sh` | Short-time variant of `scripts/leonardo/sbatch_train_libero_imf.sh` for iterating on iMF on Leonardo's debug QoS. |
