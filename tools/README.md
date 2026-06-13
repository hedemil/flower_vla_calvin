# tools/

Utilities supporting the training/evaluation flow. Not entry points themselves —
kept here for reference.

| File | Purpose |
|---|---|
| `convert_ckpt.py` | Convert a Lightning `.ckpt` checkpoint into a `.safetensors` file with the canonical `vlm.*` key layout the inference path expects. Edit the `input_ckpt` / `output_safetensors` paths at the top before running. |

The results-analysis pipeline (tables/figures from W&B + eval logs) lives in
[`tools/results/`](results/README.md).
