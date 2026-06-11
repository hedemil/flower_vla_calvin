#!/usr/bin/env python
"""Upload a from-scratch flowereef pretrain backbone (RF or iMF) to the HF Hub.

Run on the Leonardo LOGIN node (compute nodes have no internet):

    source $LEONARDO_WORK/venvs/flower_vla_calvin/bin/activate
    export HF_TOKEN=hf_xxx                 # or run: huggingface-cli login

    python scripts/leonardo/upload_pretrain_hf.py \
        --ckpt   <path to backbone>  \
        --config <path to resolved config.yaml> \
        --subfolder rf \
        --repo-id hedemil/flower-vla-flowereef-pretrain

Both pretrain backbones go into ONE public repo, one subfolder each (rf/, imf/),
each self-contained as `model.safetensors` + `config.yaml` (mirroring the layout
written by EMASafetensorsCheckpoint for the CALVIN fine-tunes).

--ckpt accepts either:
  * an Accelerate dir containing `model.safetensors`  -> copied through unchanged
  * a `.safetensors` file                             -> copied through unchanged
  * a `.pt` / `.pth` state dict                        -> converted to weights-only
    safetensors (optimizer / training state dropped; EMA weights preferred when present)

Optionally pass --model-card <path> once to upload a README.md to the repo root.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file


# Keys that commonly wrap the actual weights in a Lightning / Accelerate `.pt`.
_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")
# Prefer these (EMA) when both raw and EMA weights are present in the same file.
_EMA_KEYS = ("ema_state_dict", "ema", "ema_model", "averaged_model")


def _unwrap_state_dict(obj, prefer_ema: bool):
    """Return a flat {name: tensor} mapping from a loaded checkpoint object."""
    if not isinstance(obj, dict):
        # Already a bare state dict (e.g. an OrderedDict of tensors).
        return obj

    if prefer_ema:
        for k in _EMA_KEYS:
            if k in obj and isinstance(obj[k], dict):
                print(f"  using EMA weights from key '{k}'")
                return _unwrap_state_dict(obj[k], prefer_ema=False)

    for k in _STATE_DICT_KEYS:
        if k in obj and isinstance(obj[k], dict):
            return _unwrap_state_dict(obj[k], prefer_ema=False)

    # Heuristic: a dict whose values are all tensors is itself the state dict.
    if obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj

    sys.exit(
        "ERROR: could not locate a state dict in the checkpoint. "
        f"Top-level keys: {list(obj.keys())[:12]}"
    )


def _to_safetensors(state_dict, out_path: Path) -> None:
    """Write a state dict to safetensors, cpu + contiguous + de-aliased."""
    clean = {}
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue  # skip non-tensor bookkeeping entries
        # .clone() breaks any shared storage (safetensors rejects aliased tensors).
        clean[name] = tensor.detach().to("cpu").contiguous().clone()
    if not clean:
        sys.exit("ERROR: no tensors found to save.")
    save_file(clean, str(out_path))
    print(f"  wrote {len(clean)} tensors -> {out_path}")


def materialize_backbone(ckpt: Path, config: Path | None, prefer_ema: bool, tmp: Path) -> Path:
    """Produce a temp dir holding model.safetensors (+ config.yaml) ready to upload."""
    stage = tmp / "stage"
    stage.mkdir(parents=True, exist_ok=True)

    # 1) model.safetensors
    if ckpt.is_dir():
        src = ckpt / "model.safetensors"
        if not src.is_file():
            sys.exit(f"ERROR: {ckpt} is a dir but has no model.safetensors")
        shutil.copy2(src, stage / "model.safetensors")
        print(f"  copied {src}")
        if config is None:
            cand = ckpt / "config.yaml"
            if cand.is_file():
                config = cand
    elif ckpt.suffix == ".safetensors":
        shutil.copy2(ckpt, stage / "model.safetensors")
        print(f"  copied {ckpt}")
    elif ckpt.suffix in (".pt", ".pth", ".ckpt"):
        print(f"  loading {ckpt} (weights_only=False, cpu) ...")
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = _unwrap_state_dict(obj, prefer_ema=prefer_ema)
        _to_safetensors(sd, stage / "model.safetensors")
    else:
        sys.exit(f"ERROR: unsupported checkpoint type: {ckpt.suffix}")

    # 2) config.yaml (self-contained subfolder)
    if config is not None and Path(config).is_file():
        shutil.copy2(config, stage / "config.yaml")
        print(f"  copied config {config}")
    else:
        print("  WARNING: no config.yaml provided/found; subfolder will lack a config")

    return stage


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="backbone: Accelerate dir, .safetensors, or .pt/.pth/.ckpt")
    ap.add_argument("--config", type=Path, default=None,
                    help="resolved Hydra config.yaml to ship alongside the weights")
    ap.add_argument("--repo-id", required=True,
                    help="e.g. hedemil/flower-vla-flowereef-pretrain")
    ap.add_argument("--subfolder", required=True, choices=["rf", "imf"],
                    help="subfolder in the repo (one per backbone)")
    ap.add_argument("--private", dest="private", action="store_true", default=False,
                    help="make the repo private (default: public)")
    ap.add_argument("--no-prefer-ema", dest="prefer_ema", action="store_false", default=True,
                    help="do not prefer EMA weights when converting a .pt")
    ap.add_argument("--model-card", type=Path, default=None,
                    help="optional README.md to upload to the repo root (do once)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    if not ckpt.exists():
        sys.exit(f"ERROR: --ckpt not found: {ckpt}")

    print(f"Backbone:  {ckpt}")
    print(f"Repo:      {args.repo_id}  (private={args.private})")
    print(f"Subfolder: {args.subfolder}")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stage = materialize_backbone(ckpt, args.config, args.prefer_ema, tmp)
        api.upload_folder(
            folder_path=str(stage),
            path_in_repo=args.subfolder,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Add {args.subfolder} flowereef pretrain backbone",
            allow_patterns=["model.safetensors", "config.yaml"],
        )

        if args.model_card is not None:
            if not args.model_card.is_file():
                sys.exit(f"ERROR: --model-card not found: {args.model_card}")
            api.upload_file(
                path_or_fileobj=str(args.model_card),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message="Add model card",
            )
            print(f"  uploaded model card -> {args.repo_id}/README.md")

    print(f"\nDone. https://huggingface.co/{args.repo_id}/tree/main/{args.subfolder}")


if __name__ == "__main__":
    main()
