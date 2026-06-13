"""Experiment F: success rate vs injected inference latency (decoupled clock).

Aggregates the per-delay CALVIN runs produced by
``scripts/leonardo/sbatch_latency_sweep_calvin.sh`` (each writes a standard
``rollout_episodes.jsonl``) into a success-vs-latency curve. In normal CALVIN sim
the world pauses during inference, so latency is invisible; with
``inference_delay_steps=d`` the robot acts on a d-step-stale observation, making the
hidden cost of latency measurable.

Two comparison modes:

  * One policy (``--sweep-root``): read the cost of latency off a single curve by
    comparing the same policy at d=1 (iMF's 32.9 ms operating point) vs d=2 (the
    staleness RF's 67.0 ms would impose). Policy quality is held fixed.
  * Two policies (``--sweep imf=... --sweep rf=...``): the real deployment
    comparison -- the iMF system at its operating delay (d=1, NFE=1) vs the RF
    system at its operating delay (d=2, NFE=4). Each curve is marked with a star at
    its own operating point.

Delay -> latency map: CALVIN is 30 Hz, so d control steps ~= 33.3 ms x d. RTX 4090
operating points: iMF ~32.9 ms -> d=1, RF ~67.0 ms -> d=2.

Reads ``<sweep-root>/d<N>/logs/<timestamp>/rollout_episodes.jsonl`` (latest
timestamp per delay), where each "chain" record carries ``success_counter`` (0..5).

Outputs (to --out-dir):
  - R_calvin_sr_vs_delay.csv          : per-(variant, delay) avg chain length +
    SR-at-K (+CIs).
  - R_calvin_sr_vs_delay.{pdf,png}    : avg chain length vs d, operating points
    starred.
  - R_calvin_srK_vs_delay.{pdf,png}   : SR-at-K (K=1,3,5) vs d, one panel/variant.

Usage (from repo root):
    # single policy
    python -m tools.results.calvin_latency_sweep \
        --sweep-root <run_dir>/latency_sweep \
        --out-dir docs/figures/results/realtime

    # head-to-head (iMF system vs RF system)
    python -m tools.results.calvin_latency_sweep \
        --sweep imf=<imf_run_dir>/latency_sweep \
        --sweep rf=<rf_run_dir>/latency_sweep \
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

# Per-variant plotting + operating point: (display label, operating delay d,
# operating latency ms, color). The operating delay is where that policy actually
# sits on the latency axis on the RTX 4090.
VARIANT_INFO: dict[str, tuple[str, int, float, str]] = {
    "imf": ("iMF@1 (1-step)", 1, 32.9, style.OKABE_ITO_GREEN),
    "rf": ("RF@4 (4-step)", 2, 67.0, style.OKABE_ITO_BLUE),
}


def _variant_info(variant: str) -> tuple[str, int | None, float | None, str]:
    if variant in VARIANT_INFO:
        return VARIANT_INFO[variant]
    return (variant, None, None, "#666666")


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


def collect(sweep_root: Path, variant: str) -> pd.DataFrame:
    """One row per delay for a single sweep: avg chain length + SR-at-K with CIs."""
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
            "variant": variant,
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


def _star_operating_point(ax, sub: pd.DataFrame, variant: str) -> None:
    """Mark the variant's true operating delay with a star + latency annotation."""
    label, op_d, op_ms, color = _variant_info(variant)
    if op_d is None:
        return
    hit = sub[sub.delay_steps == op_d]
    if hit.empty:
        return
    y = float(hit.avg_chain_length.iloc[0])
    ax.scatter([op_d], [y], marker="*", s=180, color=color, edgecolor="white",
               linewidth=0.8, zorder=5)
    ax.annotate(f"{label}\n@ {op_ms:.1f} ms (d={op_d})", (op_d, y),
                textcoords="offset points", xytext=(8, 8), fontsize=7, color=color)


def plot_avg(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for variant, sub in df.groupby("variant"):
        label, _op_d, _op_ms, color = _variant_info(variant)
        ax.plot(sub.delay_steps, sub.avg_chain_length, "-o", color=color, label=label,
                zorder=3)
        ax.fill_between(sub.delay_steps, sub.avg_lo, sub.avg_hi, color=color,
                        alpha=0.13, zorder=2)
        _star_operating_point(ax, sub, variant)
    ax.set_xlabel("Injected obs→action delay d (control steps; latency ≈ 33 ms × d)")
    ax.set_ylabel("CALVIN avg. chain length (out of 5)")
    ax.set_title("Once the world stops waiting, latency costs success")
    ax.set_xticks(sorted(df.delay_steps.unique()))
    if df.variant.nunique() > 1:
        ax.legend(title="Policy (★ = its operating latency)")
    fig.tight_layout()
    fig.savefig(out_dir / "R_calvin_sr_vs_delay.pdf")
    fig.savefig(out_dir / "R_calvin_sr_vs_delay.png", dpi=200)
    plt.close(fig)


def plot_srK(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    variants = list(df.variant.unique())
    shades = {1: style.OKABE_ITO_GREEN_LIGHT, 3: style.OKABE_ITO_GREEN,
              5: style.OKABE_ITO_GREEN_DARKEST}
    fig, axes = plt.subplots(1, len(variants), figsize=(5.6 * len(variants), 4.0),
                             squeeze=False, sharey=True)
    for ax, variant in zip(axes[0], variants):
        sub = df[df.variant == variant]
        for k in (1, 3, 5):
            ax.plot(sub.delay_steps, sub[f"sr_k{k}"], "-o", color=shades[k], label=f"K≥{k}")
        label, op_d, _op_ms, color = _variant_info(variant)
        if op_d is not None:
            ax.axvline(op_d, color=color, ls=":", lw=1.1)
        ax.set_title(label)
        ax.set_xlabel("Injected delay d (control steps)")
        ax.set_xticks(sorted(sub.delay_steps.unique()))
    axes[0][0].set_ylabel("Success rate (%)")
    axes[0][0].legend(title="Tasks in a row")
    fig.suptitle("Success at chain length K degrades with latency")
    fig.tight_layout()
    fig.savefig(out_dir / "R_calvin_srK_vs_delay.pdf")
    fig.savefig(out_dir / "R_calvin_srK_vs_delay.png", dpi=200)
    plt.close(fig)


def _parse_sweeps(args) -> list[tuple[str, Path]]:
    """Resolve --sweep VARIANT=PATH (repeatable) and/or --sweep-root into pairs."""
    pairs: list[tuple[str, Path]] = []
    for spec in args.sweep or []:
        if "=" not in spec:
            raise SystemExit(f"--sweep expects VARIANT=PATH, got {spec!r}")
        variant, path = spec.split("=", 1)
        pairs.append((variant.strip(), Path(path).expanduser()))
    if args.sweep_root is not None:
        pairs.append((args.variant, args.sweep_root))
    if not pairs:
        raise SystemExit("Provide --sweep-root or one or more --sweep VARIANT=PATH")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", type=Path,
                    help="Single sweep dir containing d0/, d1/, ... (labelled by --variant).")
    ap.add_argument("--variant", default="imf",
                    help="Variant key for --sweep-root (default: imf).")
    ap.add_argument("--sweep", action="append",
                    help="VARIANT=PATH for a labelled sweep; repeat for head-to-head.")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures/results/realtime"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = [collect(path, variant) for variant, path in _parse_sweeps(args)]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out_dir / "R_calvin_sr_vs_delay.csv", index=False)
    plot_avg(df, args.out_dir)
    plot_srK(df, args.out_dir)

    print(df[["variant", "delay_steps", "latency_ms", "n_chains", "avg_chain_length",
              "sr_k1", "sr_k3", "sr_k5"]].to_string(index=False))
    print(f"\nWrote SR-vs-delay CSV + figures to {args.out_dir}")


if __name__ == "__main__":
    main()
