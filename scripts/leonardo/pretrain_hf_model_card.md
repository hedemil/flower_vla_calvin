---
license: mit
library_name: flower-vla
tags:
  - robotics
  - vision-language-action
  - flow-matching
  - meanflow
  - rectified-flow
---

# FLOWER-VLA flowereef pretrain backbones (RF and iMF)

From-scratch pretrain backbones for a FLOWER-style Vision-Language-Action (VLA) policy,
released alongside a master's thesis comparing a multi-step **Rectified Flow (RF)** action head
against a single-step **Improved MeanFlow (iMF)** action head. Both backbones share the FLOWER
architecture (Florence-2-large VLM with the text decoder removed, feeding an 18-block causally
masked DiT action expert, hidden size 1024) and differ only in the action-head objective.

Both variants were pretrained from scratch on the **same** Open-X Embodiment "flowereef" mixture
(nine public RLDS datasets, e.g. bridge, fractal, bc_z, eef_droid, dobbe, stanford_hydra,
cmu_play_fusion, libero_10, libero_goal), so RF and iMF see an identical realised data mix. They
are intended as starting points for downstream fine-tuning (e.g. LIBERO, CALVIN), not as
ready-to-deploy task policies.

## Repository layout

| Subfolder | Objective | Sampling steps (inference) | Files |
|-----------|-----------|----------------------------|-------|
| `rf/`     | Rectified Flow              | 4 | `model.safetensors`, `config.yaml` |
| `imf/`    | Improved MeanFlow           | 1 | `model.safetensors`, `config.yaml` |

Each subfolder is self-contained: weights-only `model.safetensors` plus the resolved Hydra
`config.yaml` used at pretraining time.

> Convention note: the codebase (and these configs) use the **data-to-noise** time convention
> (t=0 is data, t=1 is noise), which is the opposite of the standard flow-matching literature.

## Usage

```python
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

path = snapshot_download("hedemil/flower-vla-flowereef-pretrain")

rf_weights  = load_file(f"{path}/rf/model.safetensors")
imf_weights = load_file(f"{path}/imf/model.safetensors")
# Build the matching agent from config.yaml (Hydra) and load_state_dict(...).
```

A convenience downloader is shipped in the code repository:

```bash
scripts/download_pretrain_backbones.sh hedemil/flower-vla-flowereef-pretrain
```

## Provenance and license

Pretrained on 4xA100 GPUs (Leonardo HPC) in bfloat16. The RF arm follows the FLOWER recipe;
the iMF arm replaces the velocity head with the Improved MeanFlow objective. Released under MIT,
consistent with the upstream FLOWER codebase. See the thesis for the full pretraining mixture,
training recipe, and downstream evaluation.

## Citation

```
TODO: add thesis citation / DiVA reference once published.
```
