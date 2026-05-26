"""Action-trajectory smoothness on CALVIN rollouts, per model and per NFE.

Reads ``rollout_actions.jsonl`` files (written when ``flower_evaluate.py`` runs
with ``log_actions=true``; one ``kind="actions"`` record per attempted subtask
carrying the executed ``actions`` array ``[n_steps, action_dim]`` and the chunk
re-plan period ``multistep``). Computes, per subtask trajectory, four smoothness
measures on the commanded end-effector pose stream (first action_dim-1 dims; the
last dim is the gripper, analysed separately as a toggle count):

  1. LDLJ   — log dimensionless jerk of the speed profile (Balasubramanian 2015).
              Higher (less negative) = smoother. Duration/peak-speed normalised.
  2. SPARC  — spectral arc length of the speed profile (Balasubramanian 2012).
              Closer to 0 (less negative) = smoother; robust to duration.
  3. diff1/2 — mean L2 norm of 1st and 2nd action differences (step-to-step
              change and "acceleration" of the command). Lower = smoother.
  4. chunk  — mean ||Δa|| at chunk boundaries (steps that are multiples of
              ``multistep``) vs interior steps, and their ratio. Ratio > 1 means
              re-planning introduces a jump (jitter) at chunk boundaries.

CALVIN actions are velocity-like (relative pose deltas), so the action stream is
treated directly as the kinematic signal; the speed profile is its per-step L2
norm. The control rate ``--fs`` (default 30 Hz) only rescales LDLJ/SPARC by a
constant, so it does not affect comparisons at a fixed fs.

By default only successful subtask attempts are analysed (comparable); pass
``--include-failures`` to use all. Paired Wilcoxon tests pair runs of the same
benchmark by ``(sequence_index, subtask_position)``.

Usage::

    python -m tools.results.calvin_smoothness \\
        --run imf_dd=.../imf_dd/rollout_actions.jsonl \\
        --run flower_dd_nfe1=.../flower_dd_nfe1/rollout_actions.jsonl \\
        ... \\
        --out-dir docs/figures/results/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scstats

from . import style
from .calvin_steps_summary import BENCH_DISPLAY, parse_label


# --------------------------------------------------------------------------- #
# Smoothness primitives
# --------------------------------------------------------------------------- #

def log_dimensionless_jerk(speed: np.ndarray, fs: float) -> float:
    """LDLJ on a 1-D speed profile (Balasubramanian 2015). Higher = smoother."""
    speed = np.asarray(speed, dtype=float)
    if speed.size < 3:
        return float("nan")
    dt = 1.0 / fs
    peak = float(np.max(np.abs(speed)))
    if peak == 0.0:
        return float("nan")
    jerk = np.diff(speed, n=2) / (dt ** 2)
    integral = float(np.sum(jerk ** 2) * dt)
    if integral <= 0.0:
        return float("nan")
    dur = speed.size * dt
    scale = (dur ** 3) / (peak ** 2)
    return float(-np.log(scale * integral))


def sparc(speed: np.ndarray, fs: float, padlevel: int = 4,
          fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length of a 1-D speed profile. Closer to 0 = smoother."""
    speed = np.asarray(speed, dtype=float)
    if speed.size < 3:
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(speed.size)) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft))
    if Mf.max() == 0:
        return float("nan")
    Mf = Mf / Mf.max()
    sel = np.where(f <= fc)[0]
    f_sel, Mf_sel = f[sel], Mf[sel]
    over = np.where(Mf_sel >= amp_th)[0]
    if over.size < 2:
        return float("nan")
    f_sel = f_sel[over[0]:over[-1] + 1]
    Mf_sel = Mf_sel[over[0]:over[-1] + 1]
    span = f_sel[-1] - f_sel[0]
    if span == 0:
        return float("nan")
    return float(-np.sum(np.sqrt((np.diff(f_sel) / span) ** 2 + np.diff(Mf_sel) ** 2)))


def chunk_boundary_jumps(pose: np.ndarray, multistep: int) -> tuple[float, float, float]:
    """Mean ||Δpose|| at chunk boundaries vs interior, and their ratio.

    Boundaries are steps k (k>=1) with k % multistep == 0 — where the policy
    re-plans a fresh action chunk (the rollout resets the counter at step 0)."""
    if pose.shape[0] < 2 or multistep is None or multistep <= 1:
        return float("nan"), float("nan"), float("nan")
    jumps = np.linalg.norm(np.diff(pose, axis=0), axis=1)  # length n_steps-1, jump[k-1] = ||p[k]-p[k-1]||
    k = np.arange(1, pose.shape[0])  # step index of each jump's right endpoint
    is_boundary = (k % multistep) == 0
    boundary = jumps[is_boundary]
    interior = jumps[~is_boundary]
    if boundary.size == 0 or interior.size == 0:
        return (float(boundary.mean()) if boundary.size else float("nan"),
                float(interior.mean()) if interior.size else float("nan"),
                float("nan"))
    b, it = float(boundary.mean()), float(interior.mean())
    return b, it, (b / it if it > 0 else float("nan"))


def gripper_toggles(gripper: np.ndarray) -> int:
    """Number of open/close transitions in the gripper command stream."""
    if gripper.size < 2:
        return 0
    state = (gripper > 0).astype(int)
    return int(np.sum(np.abs(np.diff(state))))


def trajectory_metrics(actions: list, multistep: int, fs: float) -> dict:
    """Compute all smoothness metrics for one [n_steps, action_dim] trajectory."""
    arr = np.asarray(actions, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return {}
    pose = arr[:, :-1] if arr.shape[1] > 1 else arr   # all but gripper
    gripper = arr[:, -1]
    speed = np.linalg.norm(pose, axis=1)
    diffs1 = np.linalg.norm(np.diff(pose, axis=0), axis=1)
    diffs2 = np.linalg.norm(np.diff(pose, n=2, axis=0), axis=1)
    b, it, ratio = chunk_boundary_jumps(pose, multistep)
    return {
        "n_steps": int(arr.shape[0]),
        "ldlj": log_dimensionless_jerk(speed, fs),
        "sparc": sparc(speed, fs),
        "mean_diff1": float(diffs1.mean()) if diffs1.size else float("nan"),
        "mean_diff2": float(diffs2.mean()) if diffs2.size else float("nan"),
        "chunk_jump_boundary": b,
        "chunk_jump_interior": it,
        "chunk_jump_ratio": ratio,
        "gripper_toggles": gripper_toggles(gripper),
    }


# --------------------------------------------------------------------------- #
# Loading / aggregation
# --------------------------------------------------------------------------- #

METRIC_COLS = ["ldlj", "sparc", "mean_diff1", "mean_diff2",
               "chunk_jump_boundary", "chunk_jump_interior", "chunk_jump_ratio",
               "gripper_toggles"]
SMOOTHER_HIGHER = {"ldlj": True, "sparc": True}  # for these, higher = smoother


def load_action_records(path: Path, fs: float, include_failures: bool) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") != "actions":
                continue
            if not include_failures and int(rec.get("success", 0)) != 1:
                continue
            m = trajectory_metrics(rec["actions"], int(rec.get("multistep", 1)), fs)
            if not m:
                continue
            m.update({
                "sequence_index": int(rec["sequence_index"]),
                "subtask_position": int(rec["subtask_position"]),
                "subtask_name": str(rec["subtask_name"]),
                "success": int(rec.get("success", 0)),
            })
            rows.append(m)
    if not rows:
        raise SystemExit(f"No usable action trajectories in {path}")
    return pd.DataFrame(rows)


def _pick_reference(metas: list[dict]) -> str | None:
    imf = [m for m in metas if m["model"] == "imf"]
    if imf:
        return imf[0]["label"]
    flower = sorted((m for m in metas if m["nfe"] is not None), key=lambda m: m["nfe"])
    return flower[0]["label"] if flower else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                    help="Run label and its rollout_actions.jsonl. Repeatable.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fs", type=float, default=30.0, help="Control rate in Hz (CALVIN ~30).")
    ap.add_argument("--include-failures", action="store_true",
                    help="Include failed subtask attempts (default: successful only).")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metas: dict[str, dict] = {}
    per_traj: dict[str, pd.DataFrame] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        metas[label] = parse_label(label)
        df = load_action_records(Path(path), args.fs, args.include_failures)
        for key in ("label", "model", "bench", "nfe"):
            df[key] = metas[label][key]
        per_traj[label] = df

    all_traj = pd.concat(per_traj.values(), ignore_index=True)
    all_traj.to_csv(args.out_dir / "R_calvin_smoothness_pertraj.csv", index=False)

    # ---- aggregate (mean ± std over trajectories) --------------------------
    agg_rows = []
    for label, df in per_traj.items():
        row = {"label": label, "model": metas[label]["model"],
               "bench": metas[label]["bench"], "nfe": metas[label]["nfe"],
               "n_traj": int(len(df))}
        for col in METRIC_COLS:
            vals = df[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            row[f"{col}_mean"] = float(vals.mean()) if vals.size else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows).sort_values(
        ["bench", "model", "nfe"], na_position="last").reset_index(drop=True)
    agg.to_csv(args.out_dir / "R_calvin_smoothness.csv", index=False)

    # ---- paired Wilcoxon vs per-benchmark reference ------------------------
    paired_rows = []
    for bench, grp in agg.groupby("bench", dropna=False):
        ref = _pick_reference(grp.to_dict("records"))
        if ref is None:
            continue
        ref_df = per_traj[ref]
        for label in grp["label"]:
            if label == ref:
                continue
            merged = per_traj[label].merge(
                ref_df, on=["sequence_index", "subtask_position"], suffixes=("_v", "_r"))
            for col in METRIC_COLS:
                a = merged[f"{col}_v"].to_numpy(dtype=float)
                b = merged[f"{col}_r"].to_numpy(dtype=float)
                ok = ~(np.isnan(a) | np.isnan(b))
                a, b = a[ok], b[ok]
                delta = a - b
                if delta.size >= 2 and not np.all(delta == 0):
                    p = float(scstats.wilcoxon(delta, zero_method="wilcox",
                                               alternative="two-sided").pvalue)
                else:
                    p = float("nan")
                paired_rows.append({
                    "bench": bench, "label": label, "reference": ref, "metric": col,
                    "n_paired": int(delta.size), "mean_delta": float(delta.mean()) if delta.size else float("nan"),
                    "wilcoxon_p": p,
                })
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired.to_csv(args.out_dir / "R_calvin_smoothness_paired.csv", index=False)

    # ---- figure: LDLJ and chunk-jump ratio vs NFE --------------------------
    if not args.no_plot:
        import matplotlib.pyplot as plt

        style.apply_thesis_rcparams()
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
        panels = [("ldlj", "LDLJ (higher = smoother)"),
                  ("chunk_jump_ratio", "Chunk-boundary jump ratio")]
        for ax, (metric, ylabel) in zip(axes, panels):
            for bench, grp in agg.groupby("bench", dropna=False):
                bench_lbl = BENCH_DISPLAY.get(bench, str(bench))
                fl = grp[(grp["model"] == "flower") & grp["nfe"].notna()].sort_values("nfe")
                if not fl.empty:
                    ax.errorbar(fl["nfe"], fl[f"{metric}_mean"],
                                yerr=fl[f"{metric}_std"] / np.sqrt(fl["n_traj"].clip(lower=1)),
                                marker="o", capsize=2, color=style.color_for("rf"),
                                label=f"RF {bench_lbl}")
                im = grp[grp["model"] == "imf"]
                for _, ir in im.iterrows():
                    ax.scatter([ir["nfe"]], [ir[f"{metric}_mean"]], marker="*", s=120,
                               zorder=5, color=style.color_for("imf"), label=f"iMF {bench_lbl}")
            ax.set_xlabel("NFE")
            ax.set_ylabel(ylabel)
        axes[1].axhline(1.0, color="0.5", lw=0.8, ls="--")
        axes[0].legend(fontsize=7)
        fig.suptitle("Action-trajectory smoothness vs NFE")
        fig.tight_layout()
        fig_path = args.out_dir / "R_calvin_smoothness.pdf"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Wrote {fig_path}")

    print(f"Wrote {args.out_dir / 'R_calvin_smoothness.csv'}")
    print(f"Wrote {args.out_dir / 'R_calvin_smoothness_pertraj.csv'}")
    if not paired.empty:
        print(f"Wrote {args.out_dir / 'R_calvin_smoothness_paired.csv'}")
    print()
    show = ["bench", "model", "nfe", "n_traj", "ldlj_mean", "sparc_mean",
            "mean_diff1_mean", "chunk_jump_ratio_mean", "gripper_toggles_mean"]
    print(agg[show].to_string(index=False))


if __name__ == "__main__":
    main()
