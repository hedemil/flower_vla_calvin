#!/usr/bin/env python
"""Upload the best EMA checkpoint of a CALVIN fine-tune run to the HF Hub.

Run on the Leonardo LOGIN node (compute nodes have no internet):

    source $LEONARDO_WORK/venvs/flower_vla_calvin/bin/activate
    export HF_TOKEN=hf_xxx                 # or run: huggingface-cli login
    python scripts/leonardo/upload_checkpoint_hf.py <run_dir> --repo-id <user>/<repo>

<run_dir> is the Hydra run dir of a fine-tune job, e.g.
    $LEONARDO_FAST/project/flower_vla_calvin/logs/runs/2026-05-19/imf_calvin_abcd_41936895

EMASafetensorsCheckpoint writes a self-contained HF-format directory
(model.safetensors + config.yaml). That directory is uploaded into the repo
under a subfolder named after the run, so one repo can hold many runs and the
4090 side can download exactly one at a time.
"""
import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi


def find_best_ema_dir(run_dir: Path) -> Path:
    """Return the EMA checkpoint dir with the highest metric under run_dir.

    EMASafetensorsCheckpoint names dirs like
    `epoch=51_eval_lh_avg_seq_len=4.42`, so the trailing float is the metric.
    With top_k=1 only one survives, but parse defensively in case top_k>1.
    """
    cands = list(run_dir.rglob("saved_models/*/model.safetensors"))
    if not cands:
        sys.exit(
            f"ERROR: no EMA model.safetensors under {run_dir}\n"
            "       (expected saved_models/epoch=NN_..=X.XX/model.safetensors)"
        )

    def score(p: Path) -> float:
        m = re.search(r"=([0-9]+\.[0-9]+)$", p.parent.name)
        return float(m.group(1)) if m else -1.0

    return max(cands, key=score).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Hydra run dir of the fine-tune job")
    ap.add_argument("--repo-id", required=True, help="e.g. hedemil/flower-vla-calvin-ckpts")
    ap.add_argument("--subfolder", default=None, help="path in repo (default: run dir basename)")
    ap.add_argument("--public", dest="private", action="store_false", default=True,
                    help="make the repo public (default: private)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"ERROR: run_dir not found: {run_dir}")
    ema_dir = find_best_ema_dir(run_dir)
    subfolder = args.subfolder or run_dir.name

    print(f"Run dir:    {run_dir}")
    print(f"EMA ckpt:   {ema_dir}")
    print(f"            (uploading model.safetensors + config.yaml)")
    print(f"Repo:       {args.repo_id}  (private={args.private})")
    print(f"Subfolder:  {subfolder}")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ema_dir),
        path_in_repo=subfolder,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Add EMA checkpoint for {subfolder}",
        allow_patterns=["model.safetensors", "config.yaml"],
    )

    print("\nDone. On the 4090, evaluate it with:")
    print(f"  CALVIN_DATA_DIR=<path to task dir> \\")
    print(f"  scripts/rtx4090/eval_calvin_from_hf.sh {args.repo_id} {subfolder}")


if __name__ == "__main__":
    main()
