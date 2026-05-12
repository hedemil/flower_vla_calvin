"""Paired McNemar test and chain-length analysis on CALVIN rollout_episodes.jsonl.

The RolloutLongHorizon callback (modified for the thesis) writes one JSONL per
seed run dir, with two record types:
  - kind=chain   : one per sequence — sequence_index, eval_sequence list, success_counter
  - kind=subtask : one per subtask attempt — sequence_index, subtask_position, success, steps

Pairing is by (sequence_index, eval_sequence). Same seed across runs gives the
same 1000 (or N) chains in the same order, so sequence_index alone suffices
when both runs used identical config.

For each chain length L in {1..5}, this script computes:
  - per-run SR (chain completed >= L subtasks)
  - paired McNemar exact two-sided p-value across rf vs imf
  - Newcombe paired-difference CI on the diff
  - Bonferroni-corrected threshold over 5 chain lengths

Usage::

    python -m tools.results.mcnemar_calvin \\
        --rf-jsonl /path/to/flower_calvin_d_..../seed_242/rollout_episodes.jsonl \\
        --imf-jsonl /path/to/imf_calvin_d_..../seed_242/rollout_episodes.jsonl \\
        [--epoch 160]   # default: last epoch in each file
        [--out docs/figures/results/R_calvin_mcnemar.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as S


CHAIN_LENGTHS = [1, 2, 3, 4, 5]


def _load_chains(jsonl: Path, epoch: int | None = None) -> pd.DataFrame:
    """Load chain records from rollout_episodes.jsonl. If epoch is None, use the latest."""
    rows = []
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") != "chain":
            continue
        rows.append(rec)
    if not rows:
        raise SystemExit(f"No chain records found in {jsonl}")
    df = pd.DataFrame(rows)
    if epoch is None:
        epoch = int(df["epoch"].max())
    sub = df[df["epoch"] == epoch].copy()
    if sub.empty:
        epochs_present = sorted(df["epoch"].unique().tolist())
        raise SystemExit(f"No chains at epoch={epoch} in {jsonl}. Epochs present: {epochs_present}")
    return sub.reset_index(drop=True), int(epoch)


def _pair(rf_df: pd.DataFrame, imf_df: pd.DataFrame) -> pd.DataFrame:
    """Pair by sequence_index. Warn if eval_sequence mismatches (means seeds diverged)."""
    rf_df = rf_df.rename(columns={"success_counter": "rf_success_counter",
                                   "eval_sequence": "rf_eval_sequence"})
    imf_df = imf_df.rename(columns={"success_counter": "imf_success_counter",
                                     "eval_sequence": "imf_eval_sequence"})
    merged = rf_df.merge(imf_df, on="sequence_index", how="inner")
    # Sanity: same seed -> same eval_sequence
    mismatches = 0
    for _, r in merged.iterrows():
        if list(r["rf_eval_sequence"]) != list(r["imf_eval_sequence"]):
            mismatches += 1
    if mismatches:
        print(f"[warn] {mismatches}/{len(merged)} sequences differ between rf and imf — seeds may not be aligned")
    return merged


def _binary_at_chain(success_counter: pd.Series, L: int) -> np.ndarray:
    """1 if completed >= L subtasks, else 0."""
    return (success_counter.to_numpy() >= L).astype(int)


def _run_analysis(merged: pd.DataFrame, alpha: float) -> pd.DataFrame:
    n_corrections = len(CHAIN_LENGTHS)
    threshold = alpha / n_corrections
    rows = []
    for L in CHAIN_LENGTHS:
        rf_succ = _binary_at_chain(merged["rf_success_counter"], L)
        imf_succ = _binary_at_chain(merged["imf_success_counter"], L)
        table = S.mcnemar_table(imf_succ, rf_succ)  # rows = imf, cols = rf
        p = S.paired_mcnemar(imf_succ, rf_succ, exact=True)
        diff_ci = S.paired_diff_ci(imf_succ, rf_succ, alpha=alpha)
        rows.append({
            "chain_length": L,
            "n_pairs": int(len(merged)),
            "rf_sr": float(rf_succ.mean()),
            "imf_sr": float(imf_succ.mean()),
            "diff_pp": float((imf_succ.mean() - rf_succ.mean()) * 100),
            "diff_ci_lo_pp": float(diff_ci.lo * 100),
            "diff_ci_hi_pp": float(diff_ci.hi * 100),
            "n11_both_succ": int(table[0, 0]),
            "n10_imf_only": int(table[0, 1]),
            "n01_rf_only": int(table[1, 0]),
            "n00_both_fail": int(table[1, 1]),
            "p_value": float(p),
            "bonferroni_threshold": threshold,
            "significant_corr": p < threshold,
            "significant_uncorr": p < alpha,
        })
    return pd.DataFrame(rows)


def _format_table(df: pd.DataFrame, rf_epoch: int, imf_epoch: int) -> str:
    lines = []
    threshold = df["bonferroni_threshold"].iloc[0]
    lines.append(f"\n=== Paired McNemar: iMF vs RF on CALVIN  (rf_epoch={rf_epoch}, imf_epoch={imf_epoch}) ===")
    lines.append(
        f"{'chain≥L':8s} | n_pairs | rf_sr   imf_sr  Δpp   95% CI            | "
        f"n11/n10/n01/n00       | p_value     | sig"
    )
    lines.append("-" * 130)
    for _, r in df.iterrows():
        sig = "**" if r["significant_corr"] else ("*" if r["significant_uncorr"] else "")
        ci = f"[{r['diff_ci_lo_pp']:+5.1f}, {r['diff_ci_hi_pp']:+5.1f}]"
        lines.append(
            f"  ≥ {int(r['chain_length'])}   | {r['n_pairs']:7d} | "
            f"{100*r['rf_sr']:5.1f}   {100*r['imf_sr']:5.1f}   {r['diff_pp']:+5.1f}  {ci:16s} | "
            f"{r['n11_both_succ']:4d}/{r['n10_imf_only']:3d}/{r['n01_rf_only']:3d}/{r['n00_both_fail']:4d} | "
            f"{r['p_value']:.5f}    | {sig}"
        )
    # Avg seq len
    rf_avg = (df["rf_sr"].sum())  # = sum over L of SR(>=L), but careful — that's not avg seq len
    imf_avg = (df["imf_sr"].sum())
    lines.append("")
    lines.append(f"Avg sequence length:  RF = {rf_avg:.3f}   iMF = {imf_avg:.3f}   Δ = {imf_avg - rf_avg:+.3f}")
    lines.append(f"  (Table 9-normalised: RF = {100*rf_avg/5:.1f}%   iMF = {100*imf_avg/5:.1f}%)")
    lines.append(f"Bonferroni threshold (α={0.05}, 5 chain lengths): {threshold:.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rf-jsonl", type=Path, required=True,
                    help="rollout_episodes.jsonl from the RF (flower) fine-tune run")
    ap.add_argument("--imf-jsonl", type=Path, required=True,
                    help="rollout_episodes.jsonl from the iMF fine-tune run")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Eval epoch to analyse (default: latest in each file)")
    ap.add_argument("--out", type=Path, required=False)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rf_df, rf_epoch = _load_chains(args.rf_jsonl, epoch=args.epoch)
    imf_df, imf_epoch = _load_chains(args.imf_jsonl, epoch=args.epoch)
    if rf_epoch != imf_epoch:
        print(f"[warn] rf_epoch={rf_epoch} != imf_epoch={imf_epoch}; results compare different checkpoint epochs")

    merged = _pair(rf_df, imf_df)
    if len(merged) == 0:
        raise SystemExit("No paired sequences found — check sequence_index alignment.")

    results = _run_analysis(merged, args.alpha)
    print(_format_table(results, rf_epoch, imf_epoch))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.assign(rf_epoch=rf_epoch, imf_epoch=imf_epoch).to_csv(args.out, index=False)
        print(f"\nWrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
