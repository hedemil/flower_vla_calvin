"""Real-time importance of the one-step (iMF) action head vs the 4-step (RF) head.

In CALVIN simulation the policy clock and the world clock are coupled: the
simulator pauses while the network runs, so inference latency is invisible to the
success rate by construction. The contribution of the one-step head is therefore a
*strict Pareto improvement* -- equal accuracy at ~half the latency/compute -- whose
value is realized in the regimes the quasi-static simulator does not measure:
real-time closed-loop control, energy/edge deployment, and multi-robot serving.

This module renders five panels that communicate that value, all on the RTX 4090
(the eval-cluster GPU), plus a backing CSV per panel:

  A. R_realtime_pareto.{pdf,png}     -- latency vs CALVIN avg chain length; iMF@1
     matches the RF NFE-sweep accuracy at ~half the latency and dominates RF@1.
  B. R_realtime_budget.{pdf,png}     -- per-decision latency vs control-rate budgets;
     where RF crosses a budget line that iMF clears (real-time feasible / not).
  C. R_realtime_blindwindow.{pdf,png}-- one control cycle on a time axis; the shaded
     "robot acts on stale info" inference block is ~2x wider for RF.
  D. R_realtime_smoothness.{pdf,png} -- HONEST tradeoff: the one-step head is jerkier
     at chunk boundaries (open-loop stitching cost), partially mitigated by
     fix_chunk_noise. Not a win for iMF.
  E. R_realtime_throughput.{pdf,png} -- inference throughput, head forward passes,
     and a (labeled) energy-per-decision proxy.

Latency numbers: docs/figures/results/R5_latency_4090.csv (transcribed from
docs/latency_methodology.md). Accuracy numbers: tools.results.calvin_horizon.SR_AT_K
(n=1000, D->D and ABC->D). Smoothness: docs/figures/results/calvin_smoothness/
R_calvin_smoothness.csv. Self-contained; run from the repo root:

    python -m tools.results.realtime_importance --out-dir docs/figures/results/realtime
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import style
from .calvin_horizon import SR_AT_K, K as K_LIST

# Control-rate context. CALVIN runs at 30 Hz, so one control step is ~33.3 ms.
CONTROL_HZ = 30.0
CONTROL_PERIOD_MS = 1000.0 / CONTROL_HZ  # 33.3 ms

# Default repo locations for the input CSVs (overridable on the CLI).
DEFAULT_LATENCY_CSV = Path("docs/figures/results/R5_latency_4090.csv")
DEFAULT_SMOOTHNESS_CSV = Path("docs/figures/results/calvin_smoothness/R_calvin_smoothness.csv")

C_IMF = style.OKABE_ITO_GREEN
C_RF = style.OKABE_ITO_BLUE


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def avg_chain_length(bench: str, variant: str) -> float:
    """CALVIN average chain length (out of 5) = sum_k P(complete >= k subtasks)."""
    return sum(SR_AT_K[bench][variant]) / 100.0


def load_latency(path: Path) -> pd.DataFrame:
    """Wide per-variant latency table (one row per variant, ms per component)."""
    df = pd.read_csv(path)
    wide = df.pivot_table(index=["variant", "n_sampling"], columns="component",
                          values=["mean", "std"]).reset_index()
    wide.columns = ["_".join(c for c in col if c).strip("_") for col in wide.columns]
    return wide


def _lat(wide: pd.DataFrame, variant: str, comp: str = "total") -> tuple[float, float]:
    row = wide[wide.variant == variant].iloc[0]
    return float(row[f"mean_{comp}"]), float(row[f"std_{comp}"])


# --------------------------------------------------------------------------- #
# A. Pareto: latency vs accuracy
# --------------------------------------------------------------------------- #
def fig_pareto(wide: pd.DataFrame, out_dir: Path, bench: str = "dd") -> pd.DataFrame:
    """latency (x) vs avg chain length (y). RF NFE sweep + the iMF@1 point.

    RF@4 and iMF@1 totals are *measured* (R5_latency_4090.csv). RF@1/RF@2 totals
    are *derived* as VLM + N x (per-step head cost), where the per-step head cost
    is RF's measured head/NFE. Marked in the CSV `latency_source` column.
    """
    import matplotlib.pyplot as plt

    vlm_rf, _ = _lat(wide, "rf", "vlm")
    head_rf, _ = _lat(wide, "rf", "head")
    per_step_head = head_rf / 4.0  # RF head is 4 x DiT + Euler
    total_imf, std_imf = _lat(wide, "imf", "total")
    total_rf4, std_rf4 = _lat(wide, "rf", "total")

    points = [
        ("iMF@1", total_imf, std_imf, avg_chain_length(bench, "iMF@1"), C_IMF, "o", "measured"),
        ("RF@1", vlm_rf + per_step_head, np.nan, avg_chain_length(bench, "RF@1"), C_RF, "s", "derived"),
        ("RF@2", vlm_rf + 2 * per_step_head, np.nan, avg_chain_length(bench, "RF@2"), C_RF, "D", "derived"),
        ("RF@4", total_rf4, std_rf4, avg_chain_length(bench, "RF@4"), C_RF, "^", "measured"),
    ]
    tab = pd.DataFrame(
        [{"variant": v, "latency_ms": round(x, 1), "latency_std_ms": s,
          "avg_chain_length": round(y, 3), "latency_source": src}
         for v, x, s, y, _, _, src in points])

    style.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    # RF NFE-sweep frontier (light connecting line, NFE 1->2->4).
    rf_seq = [p for p in points if p[0].startswith("RF")]
    ax.plot([p[1] for p in rf_seq], [p[3] for p in rf_seq],
            color=C_RF, lw=1.0, ls="--", alpha=0.6, zorder=1)

    for v, x, s, y, c, mk, src in points:
        ax.errorbar(x, y, xerr=(None if np.isnan(s) else s), color=c, marker=mk,
                    markersize=9, capsize=3, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6)
        dy = 0.012 if v != "RF@1" else -0.022
        ax.annotate(v, (x, y), textcoords="offset points", xytext=(6, 6 if dy > 0 else -12),
                    fontsize=8)

    # Guide: iMF@1 reaches RF@4 accuracy at ~half latency.
    ax.annotate("", xy=(total_imf, avg_chain_length(bench, "iMF@1")),
                xytext=(total_rf4, avg_chain_length(bench, "iMF@1")),
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))
    ax.text((total_imf + total_rf4) / 2, avg_chain_length(bench, "iMF@1") + 0.015,
            f"{total_rf4 / total_imf:.2f}x faster\nsame accuracy", ha="center",
            fontsize=7.5, color="0.35")

    ax.set_xlabel("Inference latency per decision (ms, RTX 4090)")
    ax.set_ylabel("CALVIN avg. chain length (out of 5)")
    ax.set_title(f"Equal accuracy at half the latency ({style_bench(bench)})")
    ax.text(0.98, 0.04, "up-left is better", transform=ax.transAxes, ha="right",
            fontsize=7.5, color="0.5", style="italic")
    fig.tight_layout()
    _save(fig, out_dir, "R_realtime_pareto")
    tab.to_csv(out_dir / "R_realtime_pareto.csv", index=False)
    return tab


# --------------------------------------------------------------------------- #
# B. Latency vs control-rate budget
# --------------------------------------------------------------------------- #
def fig_budget(wide: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    variants = [("rf", "RF (4-step)", C_RF), ("imf", "iMF (1-step)", C_IMF)]
    budgets = [(10, "10 Hz"), (30, "30 Hz"), (50, "50 Hz"), (100, "100 Hz")]

    style.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ypos = np.arange(len(variants))
    rows = []
    for y, (key, lbl, c) in zip(ypos, variants):
        mean, std = _lat(wide, key, "total")
        ax.barh(y, mean, xerr=std, color=c, height=0.55, capsize=3, zorder=3)
        ax.text(mean + 2, y, f"{mean:.1f} ms", va="center", fontsize=8)
        rows.append({"variant": lbl, "latency_ms": mean, "latency_std_ms": std})

    for hz, lbl in budgets:
        xb = 1000.0 / hz
        ax.axvline(xb, color="0.45", ls=":", lw=0.9, zorder=1)
        ax.text(xb, len(variants) - 0.35, lbl, rotation=90, va="top", ha="right",
                fontsize=7, color="0.4")

    ax.set_yticks(ypos)
    ax.set_yticklabels([lbl for _, lbl, _ in variants])
    ax.set_xlabel("Per-decision latency (ms) vs control-rate budget (RTX 4090)")
    ax.set_title("If you replan every step: iMF clears budgets RF cannot")
    ax.set_xlim(0, 110)
    fig.tight_layout()
    _save(fig, out_dir, "R_realtime_budget")
    pd.DataFrame(rows).to_csv(out_dir / "R_realtime_budget.csv", index=False)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# C. Blind-window timeline
# --------------------------------------------------------------------------- #
def fig_blindwindow(wide: pd.DataFrame, out_dir: Path, eef_speed_m_s: float = 0.2) -> pd.DataFrame:
    """One control cycle per variant: shaded inference (stale) block + execution.

    Staleness distance assumes a representative end-effector speed (default
    0.2 m/s); stated in the caption as an illustrative assumption, not a measurement.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    variants = [("RF (4-step)", *_lat(wide, "rf", "total"), C_RF),
                ("iMF (1-step)", *_lat(wide, "imf", "total"), C_IMF)]

    style.apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    rows = []
    for i, (lbl, lat, _std, c) in enumerate(variants):
        y = len(variants) - 1 - i
        # Inference block: robot acting on stale observation while net runs.
        ax.barh(y, lat, left=0, height=0.5, color=c, alpha=0.35, hatch="///",
                edgecolor=c, zorder=2)
        # Execution block: fresh action applied for one control period.
        ax.barh(y, CONTROL_PERIOD_MS, left=lat, height=0.5, color=c, zorder=2)
        stale_mm = eef_speed_m_s * (lat / 1000.0) * 1000.0
        ax.text(lat / 2, y, f"think\n{lat:.0f} ms", ha="center", va="center",
                fontsize=7, color="0.15")
        ax.text(lat + CONTROL_PERIOD_MS / 2, y, "act", ha="center", va="center",
                fontsize=7, color="white")
        ax.text(lat + CONTROL_PERIOD_MS + 4, y, f"staleness ≈ {stale_mm:.1f} mm",
                va="center", fontsize=7.5, color="0.35")
        rows.append({"variant": lbl, "latency_ms": lat,
                     "staleness_mm_at_%.2fm_s" % eef_speed_m_s: round(stale_mm, 1)})

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([lbl for lbl, *_ in variants][::-1])
    ax.set_xlabel("Wall-clock time within one control cycle (ms)")
    ax.set_title(f"The blind window: world moves while the policy thinks (@{eef_speed_m_s} m/s)")
    ax.legend(handles=[Patch(facecolor="0.6", alpha=0.35, hatch="///",
                             label="inference (acting on stale info)"),
                       Patch(facecolor="0.6", label="execute fresh action")],
              loc="center right", fontsize=7)
    ax.set_xlim(0, 140)
    fig.tight_layout()
    _save(fig, out_dir, "R_realtime_blindwindow")
    pd.DataFrame(rows).to_csv(out_dir / "R_realtime_blindwindow.csv", index=False)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# D. Smoothness tradeoff (honest)
# --------------------------------------------------------------------------- #
def fig_smoothness(smooth_csv: Path, out_dir: Path, bench: str = "dd") -> pd.DataFrame:
    """Grouped bars: the one-step head is jerkier (a cost, not a win)."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(smooth_csv)
    rf = df[(df.model == "flower") & (df.bench == bench) & (df.nfe == 4)].iloc[0]
    imf = df[(df.model == "imf") & (df.bench == bench)].iloc[0]

    metrics = [("mean_diff1", "Mean step\n|Δaction|"),
               ("chunk_jump_ratio", "Chunk-boundary\njump ratio")]
    rows = []
    style.apply_thesis_rcparams()
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2, 3.2))
    for ax, (m, lbl) in zip(axes, metrics):
        vals = [rf[f"{m}_mean"], imf[f"{m}_mean"]]
        errs = [rf[f"{m}_std"], imf[f"{m}_std"]]
        ax.bar([0, 1], vals, yerr=errs, color=[C_RF, C_IMF], width=0.6, capsize=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["RF@4", "iMF@1"])
        ax.set_title(lbl)
        rows.append({"metric": m, "RF@4": round(vals[0], 4), "iMF@1": round(vals[1], 4)})
    axes[1].axhline(1.0, color="0.4", ls=":", lw=0.8)
    fig.suptitle(f"Tradeoff: the one-step head is jerkier ({style_bench(bench)}, lower is smoother)")
    fig.tight_layout()
    _save(fig, out_dir, "R_realtime_smoothness")
    pd.DataFrame(rows).to_csv(out_dir / "R_realtime_smoothness.csv", index=False)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# E. Throughput / compute proxy
# --------------------------------------------------------------------------- #
def fig_throughput(wide: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    t_rf, _ = _lat(wide, "rf", "total")
    t_imf, _ = _lat(wide, "imf", "total")
    rows = [
        {"metric": "Throughput (decisions/s)", "RF@4": 1000.0 / t_rf,
         "iMF@1": 1000.0 / t_imf, "fmt": "{:.1f}"},
        {"metric": "Head forward passes / decision", "RF@4": 4.0, "iMF@1": 1.0,
         "fmt": "{:.0f}"},
        {"metric": "Energy / decision (proxy, norm.)", "RF@4": t_rf / t_rf,
         "iMF@1": t_imf / t_rf, "fmt": "{:.2f}"},
    ]
    style.apply_thesis_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))
    for ax, r in zip(axes, rows):
        vals = [r["RF@4"], r["iMF@1"]]
        ax.bar([0, 1], vals, color=[C_RF, C_IMF], width=0.6)
        for x, v in zip([0, 1], vals):
            ax.text(x, v, r["fmt"].format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["RF@4", "iMF@1"])
        ax.set_title(r["metric"], fontsize=8)
        ax.margins(y=0.18)
    fig.suptitle("Compute side of the Pareto win (energy = latency proxy, not measured)")
    fig.tight_layout()
    _save(fig, out_dir, "R_realtime_throughput")
    out = pd.DataFrame([{k: v for k, v in r.items() if k != "fmt"} for r in rows])
    out.to_csv(out_dir / "R_realtime_throughput.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
def style_bench(bench: str) -> str:
    return {"dd": "CALVIN D→D", "abcd": "CALVIN ABC→D"}.get(bench, bench)


def _save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures/results/realtime"))
    ap.add_argument("--latency-csv", type=Path, default=DEFAULT_LATENCY_CSV)
    ap.add_argument("--smoothness-csv", type=Path, default=DEFAULT_SMOOTHNESS_CSV)
    ap.add_argument("--bench", default="dd", choices=["dd", "abcd"])
    ap.add_argument("--eef-speed", type=float, default=0.2,
                    help="Assumed end-effector speed (m/s) for the blind-window staleness annotation.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wide = load_latency(args.latency_csv)
    fig_pareto(wide, args.out_dir, args.bench)
    fig_budget(wide, args.out_dir)
    fig_blindwindow(wide, args.out_dir, args.eef_speed)
    fig_smoothness(args.smoothness_csv, args.out_dir, args.bench)
    fig_throughput(wide, args.out_dir)

    print(f"Wrote A-E figures + backing CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
