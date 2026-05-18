"""Average env steps to success per (variant, suite) on the LIBERO eval rollouts.

Reads the consolidated episode JSONL (see ``ingest_eval_dirs.py``), filters to
the requested suite and variants, and reports the mean / median / std of the
``steps`` field over *successful* episodes only. For each non-reference variant
also reports the paired mean Δsteps (vs the reference) on episodes where both
variants succeeded, plus a Wilcoxon signed-rank p-value.

Episodes without a ``steps`` field (e.g. iMF rollouts logged before the field
was added) are silently dropped per variant; the summary table flags any
variant whose successful-episode count drops to zero.

Usage::

    python -m tools.results.steps_summary \\
        --episodes-jsonl tools/results/_cache/eval_episodes.jsonl \\
        --suite libero_10 \\
        --variants rf_n1 rf_n2 rf_n3 rf \\
        --reference rf \\
        --out-dir docs/figures/results/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy import stats as scstats

from . import style


N_FROM_VARIANT = {"rf_n1": 1, "rf_n2": 2, "rf_n3": 3, "rf_n4": 4, "rf": 4, "imf": 1}


def _filter(df: pd.DataFrame, suite: str, variants: Iterable[str]) -> pd.DataFrame:
    sub = df[(df["suite"] == suite) & (df["variant"].isin(list(variants)))].copy()
    return sub.dropna(subset=["steps"])


def _summary_row(name: str, n_sampling: int | None, succ_steps: pd.Series, n_total: int) -> dict:
    n_succ = len(succ_steps)
    if n_succ == 0:
        return {
            "variant": name,
            "n_sampling": n_sampling,
            "n_success": 0,
            "n_total": n_total,
            "mean_steps": float("nan"),
            "std_steps": float("nan"),
            "median_steps": float("nan"),
        }
    return {
        "variant": name,
        "n_sampling": n_sampling,
        "n_success": n_succ,
        "n_total": n_total,
        "mean_steps": float(succ_steps.mean()),
        "std_steps": float(succ_steps.std(ddof=1)) if n_succ > 1 else 0.0,
        "median_steps": float(succ_steps.median()),
    }


def _paired_delta(df_var: pd.DataFrame, df_ref: pd.DataFrame) -> tuple[float, float, int]:
    """Paired Δsteps on episodes where both succeeded. Returns (mean Δ, p, n_paired)."""
    var_ok = df_var[df_var["success"] == 1]
    ref_ok = df_ref[df_ref["success"] == 1]
    merged = var_ok.merge(
        ref_ok, on=["task_name", "episode_index"], suffixes=("_v", "_r")
    )
    if merged.empty:
        return float("nan"), float("nan"), 0
    delta = merged["steps_v"] - merged["steps_r"]
    if (delta == 0).all() or len(delta) < 2:
        return float(delta.mean()), float("nan"), len(delta)
    res = scstats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return float(delta.mean()), float(res.pvalue), len(delta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-jsonl", type=Path, required=True)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="Variant labels to include, e.g. rf_n1 rf_n2 rf_n3 rf.")
    ap.add_argument("--reference", default="rf",
                    help="Reference variant for paired Δsteps.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.episodes_jsonl, lines=True)
    if args.reference not in args.variants:
        raise SystemExit(f"--reference {args.reference!r} must be in --variants")

    sub = _filter(df, args.suite, args.variants)
    if sub.empty:
        raise SystemExit(f"No episodes with steps for suite={args.suite}, variants={args.variants}")

    rows = []
    per_variant: dict[str, pd.DataFrame] = {}
    for v in args.variants:
        v_df = sub[sub["variant"] == v]
        per_variant[v] = v_df
        n_total = int(len(v_df))
        succ = v_df.loc[v_df["success"] == 1, "steps"]
        rows.append(_summary_row(v, N_FROM_VARIANT.get(v), succ, n_total))

    ref_df = per_variant[args.reference]
    for row in rows:
        v = row["variant"]
        if v == args.reference:
            row["paired_delta_mean"] = 0.0
            row["wilcoxon_p"] = float("nan")
            row["n_paired"] = int((ref_df["success"] == 1).sum())
        else:
            d, p, n_pair = _paired_delta(per_variant[v], ref_df)
            row["paired_delta_mean"] = d
            row["wilcoxon_p"] = p
            row["n_paired"] = n_pair

    summary = pd.DataFrame(rows)
    # Stable sort: by n_sampling, with reference last.
    summary["is_ref"] = summary["variant"] == args.reference
    summary = summary.sort_values(["is_ref", "n_sampling"]).reset_index(drop=True)
    summary = summary.drop(columns=["is_ref"])

    csv_path = args.out_dir / "R_steps_nfe.csv"
    summary.to_csv(csv_path, index=False)

    # Pretty LaTeX
    tex_path = args.out_dir / "R_steps_nfe.tex"
    lines = [
        "\\begin{tabular}{lcccrc}",
        "\\toprule",
        "Variant & $N_\\mathrm{samp}$ & $n_\\mathrm{succ}/n$ & Mean steps ($\\pm\\sigma$) & "
        "$\\overline{\\Delta}$ vs ref & Wilcoxon $p$ \\\\",
        "\\midrule",
    ]
    for r in summary.itertuples():
        if r.variant == args.reference:
            d_fmt = "--"
            p_fmt = "--"
        elif math.isnan(r.paired_delta_mean):
            d_fmt = "--"
            p_fmt = "--"
        else:
            d_fmt = f"${r.paired_delta_mean:+.1f}$"
            p_fmt = style.format_pvalue(r.wilcoxon_p) if not math.isnan(r.wilcoxon_p) else "--"
        if math.isnan(r.mean_steps):
            mean_fmt = "[no data]"
        else:
            mean_fmt = f"${r.mean_steps:.1f} \\pm {r.std_steps:.1f}$"
        n_samp_fmt = f"{int(r.n_sampling)}" if pd.notna(r.n_sampling) else "--"
        lines.append(
            f"{r.variant} & {n_samp_fmt} & "
            f"{r.n_success}/{r.n_total} & {mean_fmt} & {d_fmt} & {p_fmt} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    tex_path.write_text("\n".join(lines))

    # Box/strip plot
    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    plot_variants = [v for v in args.variants if not per_variant[v].empty]
    data = [per_variant[v].loc[per_variant[v]["success"] == 1, "steps"].to_numpy()
            for v in plot_variants]
    positions = list(range(len(plot_variants)))
    bp = ax.boxplot(
        data, positions=positions, widths=0.55, showfliers=False,
        patch_artist=True, medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, v in zip(bp["boxes"], plot_variants):
        patch.set_facecolor(style.VARIANT_COLOR.get(v, "#888"))
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")
    for i, v in enumerate(plot_variants):
        y = per_variant[v].loc[per_variant[v]["success"] == 1, "steps"].to_numpy()
        ax.scatter([i + 0.3] * len(y), y, s=4, alpha=0.5,
                   color=style.VARIANT_COLOR.get(v, "#888"))
    labels = [
        f"{v}\n(N={N_FROM_VARIANT.get(v, '?')})" for v in plot_variants
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Env steps to success")
    ax.set_title(f"Steps to success on {style.SUITE_DISPLAY.get(args.suite, args.suite)}"
                 f" (successful episodes only)")
    fig.tight_layout()
    fig_path = args.out_dir / "R_steps_nfe.pdf"
    fig.savefig(fig_path)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {fig_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
