"""Action smoothness as a function of position in the task chain (K).

Reads the per-trajectory CSV from ``calvin_smoothness.py``
(``R_calvin_smoothness_pertraj.csv``) and, for each benchmark, plots a smoothness
metric vs ``subtask_position`` for iMF@1, RF@1 and RF@4. Answers: does iMF's
(jerkier) action quality compound or stay stable deeper into a long chain?

CAVEAT (state it in the thesis): deeper positions are survivorship-biased — only
chains that succeeded through K-1 reach position K, so the population at high K
is the easier/luckier subset. Treat as exploratory, not a controlled comparison.

    python -m tools.results.calvin_smoothness_by_position \\
        --pertraj-csv docs/figures/results/calvin_smoothness/R_calvin_smoothness_pertraj.csv \\
        --metric mean_diff1 \\
        --out-dir docs/figures/results/calvin_smoothness
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import style


# (model, nfe) -> (display label, color). The matched + native references.
SELECT = [
    ("imf", 1, "iMF@1", "#009E73"),
    ("flower", 1, "RF@1", "#56B4E9"),
    ("flower", 4, "RF@4", "#0072B2"),
]
BENCH_LBL = {"dd": "CALVIN D→D", "abcd": "CALVIN ABC→D"}
METRIC_LBL = {
    "mean_diff1": "Mean per-step action change\n(higher = jerkier)",
    "mean_diff2": "Mean 2nd action difference\n(higher = jerkier)",
    "ldlj": "LDLJ (higher = smoother)",
    "sparc": "SPARC (closer to 0 = smoother)",
    "jerk_cost": "Jerk cost (higher = jerkier)",
    "tv": "Total variation (higher = jerkier)",
    "chunk_jump_ratio": "Chunk-boundary jump ratio",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pertraj-csv", type=Path, required=True)
    ap.add_argument("--metric", default="mean_diff1")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pertraj_csv)
    if args.metric not in df.columns:
        raise SystemExit(f"metric {args.metric!r} not in {sorted(df.columns)}")

    rows = []
    for bench in ["dd", "abcd"]:
        for model, nfe, label, _ in SELECT:
            sel = df[(df["bench"] == bench) & (df["model"] == model) & (df["nfe"] == nfe)]
            for pos, g in sel.groupby("subtask_position"):
                vals = g[args.metric].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                rows.append({
                    "bench": bench, "variant": label, "K": int(pos) + 1, "n": int(vals.size),
                    "mean": float(vals.mean()),
                    "sem": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No rows — check that the pertraj CSV has the expected "
                         "model/nfe/bench/subtask_position columns.")
    out.to_csv(args.out_dir / f"R_calvin_{args.metric}_by_K.csv", index=False)

    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, bench in zip(axes, ["dd", "abcd"]):
        for _, _, label, color in SELECT:
            s = out[(out["bench"] == bench) & (out["variant"] == label)].sort_values("K")
            if s.empty:
                continue
            ax.errorbar(s["K"], s["mean"], yerr=s["sem"], marker="o", capsize=2,
                        color=color, label=label)
        ax.set_title(BENCH_LBL[bench])
        ax.set_xlabel("Position in chain (K)")
        ax.set_xticks([1, 2, 3, 4, 5])
    axes[0].set_ylabel(METRIC_LBL.get(args.metric, args.metric))
    axes[0].legend()
    fig.suptitle(f"Action smoothness vs chain position ({args.metric}; survivorship-biased)")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"R_calvin_{args.metric}_by_K.pdf")
    fig.savefig(args.out_dir / f"R_calvin_{args.metric}_by_K.png", dpi=200)
    plt.close(fig)

    print(out.to_string(index=False))
    print(f"\nWrote to {args.out_dir}")


if __name__ == "__main__":
    main()
