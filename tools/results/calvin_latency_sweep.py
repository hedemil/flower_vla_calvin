"""Experiment F: success rate vs injected inference latency (decoupled clock).

Aggregates the per-delay CALVIN runs produced by
``scripts/leonardo/sbatch_latency_sweep_calvin.sh`` (each writes a standard
``rollout_episodes.jsonl``) into a success-vs-latency curve. In normal CALVIN sim
the world pauses during inference, so latency is invisible; with
``inference_delay_steps=d`` the robot acts on a d-step-stale observation, making the
hidden cost of latency measurable. The 1-step iMF head and 4-step RF head sit at
different d (RTX 4090: iMF d=1 at 32.9 ms, RF d=2 at 67.0 ms), so the curve converts
the measured latency saving into a success-rate margin.

Reads ``<sweep-root>/d<N>/logs/<timestamp>/rollout_episodes.jsonl`` (latest
timestamp per delay), where each "chain" record carries ``success_counter`` (0..5).

Outputs (to --out-dir):
  - R_calvin_sr_vs_delay.csv          : per-delay avg chain length + SR-at-K (+CIs).
  - R_calvin_sr_vs_delay.{pdf,png}    : avg chain length vs d, with the iMF/RF
    operating-point delays marked.
  - R_calvin_srK_vs_delay.{pdf,png}   : SR-at-K (K=1,3,5) vs d.

Usage (from repo root):
    python -m tools.results.calvin_latency_sweep \
        --sweep-root <run_dir>/latency_sweep \
        --out-dir docs/figures/results/realtime
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import style
from .stats import clopper_pearson

# Control rate -> latency mapping context (CALVIN is 30 Hz).
CONTROL_PERIOD_MS = 1000.0 / 30.0
# Operating points to annotate: (label, delay_steps, approx latency ms, color).
OPERATING_POINTS = [
    ("iMF (1-step), 32.9 ms", 1, style.OKABE_ITO_GREEN),
    ("RF (4-step), 67.0 ms", 2, style.OKABE_ITO_BLUE),
]


def _latest_episodes(delay_dir: Path) -> Path | None:
    cands = sorted(delay_dir.glob("logs/*/rollout_episodes.jsonl"))
    if not cands:
        cands = sorted(delay_dir.glob("**/rollout_episodes.jsonl"))
    return cands[-1] if cands else None


def _chain_counters(jsonl: Path) -> list[int]:
    out = []
    with open(jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "chain":
                out.append(int(rec["success_counter"]))
    return out


def collect(sweep_root: Path) -> pd.DataFrame:
    """One row per delay: avg chain length + SR-at-K with CIs."""
    rows = []
    delay_dirs = sorted(sweep_root.glob("d*"),
                        key=lambda p: int(re.sub(r"\D", "", p.name) or -1))
    for ddir in delay_dirs:
        m = re.search(r"d(\d+)", ddir.name)
        if not m:
            continue
        d = int(m.group(1))
        jsonl = _latest_episodes(ddir)
        if jsonl is None:
            print(f"WARN: no rollout_episodes.jsonl under {ddir}, skipping")
            continue
        counters = np.array(_chain_counters(jsonl), dtype=int)
        n = len(counters)
        if n == 0:
            print(f"WARN: empty episodes file {jsonl}, skipping")
            continue
        avg = float(counters.mean())
        sem = float(counters.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        row = {
            "delay_steps": d,
            "latency_ms": round(d * CONTROL_PERIOD_MS, 1),
            "n_chains": n,
            "avg_chain_length": round(avg, 4),
            "avg_lo": round(avg - 1.96 * sem, 4),
            "avg_hi": round(avg + 1.96 * sem, 4),
        }
        for k in range(1, 6):
            n_succ = int((counters >= k).sum())
            ci = clopper_pearson(n_succ, n)
            row[f"sr_k{k}"] = round(100 * ci.point, 2)
            row[f"sr_k{k}_lo"] = round(100 * ci.lo, 2)
            row[f"sr_k{k}_hi"] = round(100 * ci.hi, 2)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No usable delay runs found under {sweep_root}")
    return pd.DataFrame(rows).sort_values("delay_steps").reset_index(drop=True)


def _mark_operating_points(ax) -> None:
    for lbl, d, c in OPERATING_POINTS:
        ax.axvline(d, color=c, ls=":", lw=1.1, zorder=1)
        ax.text(d, ax.get_ylim()[1], lbl, rotation=90, va="top", ha="right",
                fontsize=7, color=c)


def plot_avg(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(df.delay_steps, df.avg_chain_length, "-o", color="0.2", zorder=3)
    ax.fill_between(df.delay_steps, df.avg_lo, df.avg_hi, color="0.2", alpha=0.15, zorder=2)
    _mark_operating_points(ax)
    ax.set_xlabel("Injected obs→action delay d (control steps; latency ≈ 33 ms × d)")
    ax.set_ylabel("CALVIN avg. chain length (out of 5)")
    ax.set_title("Once the world stops waiting, latency costs success")
    ax.set_xticks(df.delay_steps)
    fig.tight_layout()
    fig.savefig(out_dir / "R_calvin_sr_vs_delay.pdf")
    fig.savefig(out_dir / "R_calvin_sr_vs_delay.png", dpi=200)
    plt.close(fig)


def plot_srK(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    shades = {1: style.OKABE_ITO_GREEN_LIGHT, 3: style.OKABE_ITO_GREEN, 5: style.OKABE_ITO_GREEN_DARKEST}
    for k in (1, 3, 5):
        ax.plot(df.delay_steps, df[f"sr_k{k}"], "-o", color=shades[k], label=f"K≥{k}")
    _mark_operating_points(ax)
    ax.set_xlabel("Injected obs→action delay d (control steps)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success at chain length K degrades with latency")
    ax.set_xticks(df.delay_steps)
    ax.legend(title="Tasks in a row")
    fig.tight_layout()
    fig.savefig(out_dir / "R_calvin_srK_vs_delay.pdf")
    fig.savefig(out_dir / "R_calvin_srK_vs_delay.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", type=Path, required=True,
                    help="Directory containing d0/, d1/, ... from the sweep script.")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures/results/realtime"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = collect(args.sweep_root)
    df.to_csv(args.out_dir / "R_calvin_sr_vs_delay.csv", index=False)
    plot_avg(df, args.out_dir)
    plot_srK(df, args.out_dir)

    print(df[["delay_steps", "latency_ms", "n_chains", "avg_chain_length",
              "sr_k1", "sr_k3", "sr_k5"]].to_string(index=False))
    print(f"\nWrote SR-vs-delay CSV + figures to {args.out_dir}")


if __name__ == "__main__":
    main()
