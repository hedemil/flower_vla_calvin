"""Paired McNemar test on consolidated LIBERO eval episodes.

For each suite, pairs RF vs iMF (and optionally RF cross-NFE variants vs iMF)
by (task_name, episode_index) and reports the McNemar exact two-sided p-value
along with the 2x2 contingency table. Bonferroni-corrects across suites.

Usage::

    python -m tools.results.mcnemar_libero \\
        --episodes tools/results/_cache/eval_episodes.jsonl \\
        --out docs/figures/results/R4_mcnemar_libero.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as S


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _load(jsonl: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(jsonl) if l.strip()]
    return pd.DataFrame(rows)


def _pair(df: pd.DataFrame, suite: str, var_a: str, var_b: str) -> pd.DataFrame | None:
    a = df[(df.suite == suite) & (df.variant == var_a)]
    b = df[(df.suite == suite) & (df.variant == var_b)]
    if a.empty or b.empty:
        return None
    merged = a.merge(
        b[["task_name", "episode_index", "success", "steps"]],
        on=["task_name", "episode_index"],
        suffixes=("_a", "_b"),
        how="inner",
    )
    return merged if not merged.empty else None


def _run_comparison(df: pd.DataFrame, var_a: str, var_b: str, alpha: float) -> pd.DataFrame:
    rows = []
    for suite in SUITES:
        merged = _pair(df, suite, var_a, var_b)
        if merged is None:
            rows.append({
                "suite": suite, "var_a": var_a, "var_b": var_b,
                "n_pairs": 0, "n_a": 0, "n_b": 0,
                "sr_a": np.nan, "sr_b": np.nan, "diff_pp": np.nan,
                "n11": 0, "n10": 0, "n01": 0, "n00": 0,
                "p_value": np.nan,
            })
            continue
        a = merged["success_a"].astype(int).to_numpy()
        b = merged["success_b"].astype(int).to_numpy()
        table = S.mcnemar_table(a, b)
        p = S.paired_mcnemar(a, b, exact=True)
        rows.append({
            "suite": suite,
            "var_a": var_a, "var_b": var_b,
            "n_pairs": int(len(merged)),
            "n_a": int(a.sum()),
            "n_b": int(b.sum()),
            "sr_a": float(a.mean()),
            "sr_b": float(b.mean()),
            "diff_pp": float((a.mean() - b.mean()) * 100),
            "n11": int(table[0, 0]),  # both succeed
            "n10": int(table[0, 1]),  # a only
            "n01": int(table[1, 0]),  # b only
            "n00": int(table[1, 1]),  # both fail
            "p_value": p,
        })
    return pd.DataFrame(rows)


def _annotate_significance(df: pd.DataFrame, alpha: float, n_corrections: int) -> pd.DataFrame:
    threshold = alpha / n_corrections
    df = df.copy()
    df["bonferroni_threshold"] = threshold
    df["significant"] = df["p_value"] < threshold
    return df


def _format_table(df: pd.DataFrame) -> str:
    lines = []
    var_a, var_b = df["var_a"].iloc[0], df["var_b"].iloc[0]
    lines.append(f"\n=== {var_a} vs {var_b} ===")
    lines.append(
        f"{'suite':16s} | n_pairs |  k_a /  k_b |  sr_a%  sr_b%  Δpp  | "
        f"n11/n10/n01/n00 | p_value     | sig"
    )
    lines.append("-" * 110)
    threshold = df["bonferroni_threshold"].iloc[0]
    for _, r in df.iterrows():
        sig = "**" if r["significant"] else ("*" if r["p_value"] < 0.05 else "")
        p_str = "—" if np.isnan(r["p_value"]) else f"{r['p_value']:.4f}"
        sr_a = "—" if np.isnan(r["sr_a"]) else f"{100*r['sr_a']:5.1f}"
        sr_b = "—" if np.isnan(r["sr_b"]) else f"{100*r['sr_b']:5.1f}"
        diff = "—" if np.isnan(r["diff_pp"]) else f"{r['diff_pp']:+5.1f}"
        lines.append(
            f"{r['suite']:16s} | {r['n_pairs']:7d} | {r['n_a']:4d} / {r['n_b']:4d} | "
            f"{sr_a}  {sr_b}  {diff} | "
            f"{r['n11']:3d}/{r['n10']:3d}/{r['n01']:3d}/{r['n00']:3d} | "
            f"{p_str:11s} | {sig}"
        )
    lines.append(f"Bonferroni threshold (α=0.05, 4 suites): {threshold:.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--episodes",
        type=Path,
        default=Path("tools/results/_cache/eval_episodes.jsonl"),
    )
    ap.add_argument("--out", type=Path, required=False)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--include-cross-nfe",
        action="store_true",
        help="Also compare iMF vs RF cross-NFE variants (rf_n1, rf_n2, rf_n3).",
    )
    args = ap.parse_args()

    df = _load(args.episodes)

    # Primary comparison: iMF vs RF at default NFE.
    n_corrections = len(SUITES)
    primary = _run_comparison(df, "imf", "rf", args.alpha)
    primary = _annotate_significance(primary, args.alpha, n_corrections)
    print(_format_table(primary))

    all_results = [primary]

    if args.include_cross_nfe:
        # Matched-compute claim: iMF (NFE=1) vs RF at NFE=1/2/3.
        # Cross-NFE data only exists for libero_10 (per ingest_eval_dirs).
        for rf_var in ["rf_n1", "rf_n2", "rf_n3"]:
            cmp = _run_comparison(df, "imf", rf_var, args.alpha)
            cmp = _annotate_significance(cmp, args.alpha, n_corrections)
            print(_format_table(cmp))
            all_results.append(cmp)

    if args.out is not None:
        out_df = pd.concat(all_results, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"\nWrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
