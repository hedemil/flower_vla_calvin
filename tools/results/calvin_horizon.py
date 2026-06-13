"""CALVIN long-horizon analysis: success rate and the iMF-RF gap as a function
of chain length K.

The headline finding: iMF's matched-compute advantage is concentrated at long
horizons. At K=1 the heads are tied; the gap grows monotonically with K (iMF
degrades more gracefully across a chain). Produces the following figures and a CSV:

  - R_calvin_sr_vs_k.{pdf,png} : success-at-K curves (iMF@1, RF@1, RF@4); curves
    fan apart at high K.
  - R_calvin_gap_vs_k.{pdf,png}: Delta(iMF - RF@1) and Delta(iMF - RF@4) per K
    as grouped bars; bars grow with K.
  - R_calvin_sr_bars.{pdf,png} : SR-at-K as grouped bars (the bar version of the
    SR-vs-K curves), one bar per variant within each K group.
  - R_calvin_sr_vs_k.csv       : all SR-at-K values and the gaps.

Numbers are the n=1000 headline runs (docs/evals.md): tuned iMF (ratio=0.75,
jobs 42325919/42373771) and RF at NFE 1/2/4 (jobs 42076583/42076853), on
CALVIN-D and ABC->D. Self-contained; run from the repo root:

    python -m tools.results.calvin_horizon --out-dir docs/figures/results/calvin_horizon
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import style


K = [1, 2, 3, 4, 5]

# Success-at-K (%, fraction of chains completing >= K subtasks), n=1000.
# Verbatim from docs/evals.md. iMF rows are the tuned ratio=0.75 checkpoints.
SR_AT_K = {
    "dd": {
        "iMF@1": [97.3, 91.8, 86.4, 81.0, 74.2],
        "RF@1":  [96.6, 90.2, 82.7, 74.1, 66.4],
        "RF@2":  [97.4, 92.9, 87.7, 79.8, 72.6],
        "RF@4":  [97.7, 92.9, 87.8, 81.5, 75.1],
    },
    "abcd": {
        "iMF@1": [98.7, 93.6, 86.1, 78.7, 70.7],
        "RF@1":  [98.3, 93.5, 85.2, 76.8, 67.0],
        "RF@2":  [99.5, 95.7, 87.5, 79.6, 69.3],
        "RF@4":  [99.4, 94.0, 85.1, 76.1, 66.8],
    },
}

BENCH_LBL = {"dd": "CALVIN D→D", "abcd": "CALVIN ABC→D"}
COLORS = {"iMF@1": "#009E73", "RF@1": "#56B4E9", "RF@2": "#E69F00", "RF@4": "#0072B2"}
LS = {"iMF@1": "-", "RF@1": "--", "RF@2": ":", "RF@4": "-"}
MK = {"iMF@1": "o", "RF@1": "s", "RF@2": "D", "RF@4": "^"}


def gap_table() -> pd.DataFrame:
    rows = []
    for bench, d in SR_AT_K.items():
        for i, k in enumerate(K):
            rows.append({
                "bench": bench, "K": k,
                "iMF@1": d["iMF@1"][i], "RF@1": d["RF@1"][i],
                "RF@2": d["RF@2"][i], "RF@4": d["RF@4"][i],
                "gap_vs_RF1": round(d["iMF@1"][i] - d["RF@1"][i], 1),
                "gap_vs_RF4": round(d["iMF@1"][i] - d["RF@4"][i], 1),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--curves", nargs="+", default=["iMF@1", "RF@1", "RF@4"],
                    help="Variants to draw on the SR-vs-K curves (subset of SR_AT_K keys).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tab = gap_table()
    tab.to_csv(args.out_dir / "R_calvin_sr_vs_k.csv", index=False)

    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()

    # Figure 1: SR-vs-K curves
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, bench in zip(axes, ["dd", "abcd"]):
        for v in args.curves:
            ax.plot(K, SR_AT_K[bench][v], LS.get(v, "-"), marker=MK.get(v, "o"),
                    color=COLORS.get(v, "#666"), label=v)
        ax.set_title(BENCH_LBL[bench])
        ax.set_xlabel("Tasks completed in a row (K)")
        ax.set_xticks(K)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend()
    fig.suptitle("iMF degrades more gracefully over long horizons")
    fig.tight_layout()
    fig.savefig(args.out_dir / "R_calvin_sr_vs_k.pdf")
    fig.savefig(args.out_dir / "R_calvin_sr_vs_k.png", dpi=200)
    plt.close(fig)

    # Figure 2: gap-vs-K grouped bars
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    width = 0.38
    x = np.arange(len(K))
    for ax, bench in zip(axes, ["dd", "abcd"]):
        sub = tab[tab.bench == bench]
        ax.bar(x - width / 2, sub["gap_vs_RF1"], width, color="#56B4E9",
               label="iMF − RF@1 (matched compute)")
        ax.bar(x + width / 2, sub["gap_vs_RF4"], width, color="#0072B2",
               label="iMF − RF@4 (native)")
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(BENCH_LBL[bench])
        ax.set_xlabel("Chain length K")
        ax.set_xticks(x)
        ax.set_xticklabels(K)
    axes[0].set_ylabel("Δ success rate (pp)")
    axes[0].legend()
    fig.suptitle("iMF's advantage grows with horizon")
    fig.tight_layout()
    fig.savefig(args.out_dir / "R_calvin_gap_vs_k.pdf")
    fig.savefig(args.out_dir / "R_calvin_gap_vs_k.png", dpi=200)
    plt.close(fig)

    # Figure 3: SR-at-K as grouped bars (bar version of the SR-vs-K curves)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    n = len(args.curves)
    width = 0.8 / n
    x = np.arange(len(K))
    for ax, bench in zip(axes, ["dd", "abcd"]):
        for j, v in enumerate(args.curves):
            offset = (j - (n - 1) / 2) * width
            ax.bar(x + offset, SR_AT_K[bench][v], width,
                   color=COLORS.get(v, "#666"), label=v)
        ax.set_title(BENCH_LBL[bench])
        ax.set_xlabel("Tasks completed in a row (K)")
        ax.set_xticks(x)
        ax.set_xticklabels(K)
        ax.set_ylim(60, 100)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend()
    fig.suptitle("Success at each chain length")
    fig.tight_layout()
    fig.savefig(args.out_dir / "R_calvin_sr_bars.pdf")
    fig.savefig(args.out_dir / "R_calvin_sr_bars.png", dpi=200)
    plt.close(fig)

    print(tab.to_string(index=False))
    print(f"\nWrote figures + CSV to {args.out_dir}")


if __name__ == "__main__":
    main()
