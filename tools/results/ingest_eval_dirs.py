"""Consolidate LIBERO eval JSONLs from `scripts/evaluation/` into one tagged file.

Walks `scripts/evaluation/libero_<suite>_<variant>_evaluation/logs/<timestamp>/episodes.jsonl`,
picks the most-recent timestamp per directory, remaps the per-record `variant`
field to the canonical label expected by `analyze_libero_results.py`
(`flower` -> `rf`, `flower_nfe<N>` -> `rf_n<N>`, `imf*` kept as-is), and emits a
single consolidated JSONL.

Usage::

    python -m tools.results.ingest_eval_dirs \
        --eval-root scripts/evaluation \
        --out tools/results/_cache/eval_episodes.jsonl \
        --skip libero_10_flower_evaluation

The `--skip` flag is for directories where the rollouts are known-bad (e.g.,
wrong checkpoint) and should be excluded entirely until rerun.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VARIANT_REMAP = {
    "flower": "rf",
    "flower_nfe1": "rf_n1",
    "flower_nfe2": "rf_n2",
    "flower_nfe3": "rf_n3",
    "flower_nfe4": "rf",
    # cross-NFE: same baseline RF ckpt, inference NFE varied at eval time.
    "flower_cross_nfe1": "rf_n1",
    "flower_cross_nfe2": "rf_n2",
    "flower_cross_nfe3": "rf_n3",
    "flower_cross_nfe4": "rf",
}


def canonical_variant(raw: str) -> str:
    return VARIANT_REMAP.get(raw, raw)


def latest_jsonl(eval_dir: Path) -> Path | None:
    candidates = sorted(eval_dir.glob("logs/*/episodes.jsonl"))
    return candidates[-1] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, default=Path("scripts/evaluation"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Evaluation directory names to skip (repeat for multiple).",
    )
    ap.add_argument(
        "--include-ablations",
        action="store_true",
        help="Include imf_ratio / imf_heads / imf_both ablation directories.",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^libero_(?P<suite>[a-z0-9]+)_(?P<variant>[a-z0-9_]+)_evaluation$")
    skip = set(args.skip)
    total = 0
    summary: list[tuple[str, str, int, int]] = []

    with open(args.out, "w") as out_f:
        for eval_dir in sorted(args.eval_root.glob("libero_*_evaluation")):
            if eval_dir.name in skip:
                print(f"[skip] {eval_dir.name}")
                continue
            m = pattern.match(eval_dir.name)
            if not m:
                print(f"[warn] unrecognized dir name: {eval_dir.name}")
                continue
            raw_variant = m.group("variant")
            if not args.include_ablations and raw_variant.startswith("imf_"):
                continue
            jsonl = latest_jsonl(eval_dir)
            if jsonl is None:
                print(f"[warn] no episodes.jsonl in {eval_dir.name}")
                continue
            n = k = 0
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec["variant"] = canonical_variant(rec.get("variant", raw_variant))
                    rec.setdefault("suite", f"libero_{m.group('suite')}")
                    out_f.write(json.dumps(rec) + "\n")
                    n += 1
                    k += int(bool(rec["success"]))
            total += n
            summary.append((eval_dir.name, canonical_variant(raw_variant), k, n))

    print(f"\nWrote {total} episodes to {args.out}")
    print(f"{'directory':50s}  {'variant':10s}  k/n   sr%")
    for name, variant, k, n in summary:
        print(f"{name:50s}  {variant:10s}  {k}/{n}  {100*k/n:5.1f}")


if __name__ == "__main__":
    main()
