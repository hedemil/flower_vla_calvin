"""Produce the RF NFE-sweep table and figure for §6.4 of the Results chapter.

Reads the consolidated episode JSONL (one record per rollout, with `variant`,
`suite`, `task_name`, `episode_index`, `success`), filters to RF NFE-sweep
variants on a chosen suite (default: libero_10 / Long-10), and emits:

* ``R_NFE_sweep.{csv,tex}`` -- one row per RF NFE point plus an iMF reference
  row, columns ``[variant, n_sampling, sr_pp, ci_lo, ci_hi]``.
* ``R_NFE_sweep.pdf`` -- single-panel SR-vs-N figure with shaded CP band and
  a horizontal dashed line for iMF@1.

Variant naming convention (set by ``ingest_eval_dirs.py``):

* ``rf_n1, rf_n2, rf_n3``: explicit num_sampling_steps overrides
* ``rf``: the default RF run (N_sampling = 4 per ``conf/eval_libero.yaml``)
* ``imf``: single-step iMF, used as a horizontal reference

Usage::

    python -m tools.results.nfe_sweep \\
        --episodes-jsonl tools/results/_cache/eval_episodes.jsonl \\
        --suite libero_10 \\
        --out-dir docs/figures/results/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats as S
from . import style


# variant label -> N_sampling for plotting / table x-axis
N_FROM_VARIANT = {
    "rf_n1": 1,
    "rf_n2": 2,
    "rf_n3": 3,
    "rf_n4": 4,
    "rf": 4,  # default num_sampling_steps from conf/eval_libero.yaml
}


def task_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Per-task CP CI for one (variant, suite) cell."""
    rows = []
    for (variant, suite, task), g in df.groupby(["variant", "suite", "task_name"]):
        k = int(g["success"].sum())
        n = int(len(g))
        ci = S.clopper_pearson(k, n)
        rows.append({
            "variant": variant,
            "suite": suite,
            "task_name": task,
            "k": k,
            "n": n,
            "sr": ci.point,
            "ci_lo": ci.lo,
            "ci_hi": ci.hi,
        })
    return pd.DataFrame(rows)


def suite_ci(per_task: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, suite), g in per_task.groupby(["variant", "suite"]):
        intervals = [S.CI(point=r.sr, lo=r.ci_lo, hi=r.ci_hi) for r in g.itertuples()]
        avg = S.average_ci(intervals)
        rows.append({
            "variant": variant,
            "suite": suite,
            "sr": avg.point,
            "ci_lo": avg.lo,
            "ci_hi": avg.hi,
            "n_tasks": len(g),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-jsonl", type=Path, required=True)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--imf-variant", default="imf",
                    help="Variant label to use as the iMF@1 reference line.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.episodes_jsonl, lines=True)
    df = df[df["suite"] == args.suite].copy()
    if df.empty:
        raise SystemExit(f"No episodes for suite={args.suite!r}")

    keep = set(N_FROM_VARIANT) | {args.imf_variant}
    df = df[df["variant"].isin(keep)]
    if df.empty:
        raise SystemExit(f"No RF/iMF variants in suite={args.suite!r}")

    per_task = task_ci(df)
    per_suite = suite_ci(per_task)

    # Attach N_sampling and sort
    per_suite["n_sampling"] = per_suite["variant"].map(
        lambda v: N_FROM_VARIANT.get(v, 1 if v == args.imf_variant else None)
    )
    per_suite["is_reference"] = per_suite["variant"] == args.imf_variant
    per_suite = per_suite.sort_values(["is_reference", "n_sampling"]).reset_index(drop=True)

    # --- table ---
    tab = per_suite[["variant", "n_sampling", "sr", "ci_lo", "ci_hi"]].copy()
    tab["sr_pp"] = (tab["sr"] * 100).round(1)
    tab["ci_lo_pp"] = (tab["ci_lo"] * 100).round(1)
    tab["ci_hi_pp"] = (tab["ci_hi"] * 100).round(1)
    tab["display"] = tab.apply(
        lambda r: f"{r.sr_pp:.1f}\\,[{r.ci_lo_pp:.1f},{r.ci_hi_pp:.1f}]", axis=1
    )
    tab_out = tab[["variant", "n_sampling", "sr_pp", "ci_lo_pp", "ci_hi_pp", "display"]]
    csv_path = args.out_dir / "R_NFE_sweep.csv"
    tex_path = args.out_dir / "R_NFE_sweep.tex"
    tab_out.to_csv(csv_path, index=False)

    # Hand-rolled TeX, since the layout expected by the thesis is small.
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Variant & $N_\\mathrm{sampling}$ & Long-10 $\\bar{S}$\\,[CI] \\\\",
        "\\midrule",
    ]
    rf_rows = per_suite[~per_suite.is_reference].sort_values("n_sampling")
    for r in rf_rows.itertuples():
        lines.append(
            f"RF       & {int(r.n_sampling)} & "
            f"${r.sr*100:.1f}$\\,[${r.ci_lo*100:.1f}$,${r.ci_hi*100:.1f}$] \\\\"
        )
    lines.append("\\midrule")
    ref = per_suite[per_suite.is_reference].iloc[0]
    lines.append(
        f"iMF (ref) & {int(ref.n_sampling)} & "
        f"${ref.sr*100:.1f}$\\,[${ref.ci_lo*100:.1f}$,${ref.ci_hi*100:.1f}$] \\\\"
    )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    tex_path.write_text("\n".join(lines))

    # --- figure ---
    style.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    rf_pts = per_suite[~per_suite.is_reference].sort_values("n_sampling")
    xs = rf_pts["n_sampling"].astype(int).to_numpy()
    ys = (rf_pts["sr"] * 100).to_numpy()
    lo = (rf_pts["ci_lo"] * 100).to_numpy()
    hi = (rf_pts["ci_hi"] * 100).to_numpy()

    rf_color = style.VARIANT_COLOR.get("rf", "#0072B2")
    imf_color = style.VARIANT_COLOR.get("imf", "#009E73")

    ax.fill_between(xs, lo, hi, color=rf_color, alpha=0.15, linewidth=0)
    ax.plot(xs, ys, "o-", color=rf_color, linewidth=1.8, markersize=6, label="RF")

    ref_sr = float(ref["sr"]) * 100
    ref_lo = float(ref["ci_lo"]) * 100
    ref_hi = float(ref["ci_hi"]) * 100
    ax.axhspan(ref_lo, ref_hi, color=imf_color, alpha=0.10, linewidth=0)
    ax.axhline(ref_sr, color=imf_color, linestyle="--", linewidth=1.5, label=f"iMF@1 ({ref_sr:.1f}\\%)")

    ax.set_xticks(sorted(xs))
    ax.set_xlabel("$N_\\mathrm{sampling}$")
    ax.set_ylabel("Suite success rate $\\bar{S}$ (\\%)")
    ax.set_ylim(-3, 102)
    ax.set_title(f"RF NFE sweep on {args.suite.replace('libero_', 'LIBERO-')}")
    ax.legend(loc="center right", frameon=False)
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig_path = args.out_dir / "R_NFE_sweep.pdf"
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {fig_path}")
    print()
    print(per_suite[["variant", "n_sampling", "sr", "ci_lo", "ci_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
