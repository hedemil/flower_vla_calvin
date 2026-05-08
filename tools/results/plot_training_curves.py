"""Pretraining-curve plots and stability summary
(`docs/thesis_results.md` §3, Figs R1/R2/R3 + Table R1).

Inputs (one of):
  --manifest manifest.json     JSON list with {csv, variant?, config?} entries.
  --csv  path:variant ...      Inline list of (csv_path, variant_label) pairs.

Outputs (in --out-dir):
  R1_loss_curves.pdf
  R2_head_diagnostics.pdf
  R3_val_loss_per_dataset.pdf
  R1_stability_summary.{csv,tex}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import stats as S
from . import style
from . import wandb_io


LOG = logging.getLogger("plot_training_curves")


# ----- Loading --------------------------------------------------------------

def load_manifest(path: str | Path) -> pd.DataFrame:
    return wandb_io.load_run_history_manifest(path)


def load_inline(pairs: list[str]) -> pd.DataFrame:
    """Each pair is `path:variant_label`."""
    entries = []
    for p in pairs:
        if ":" not in p:
            raise SystemExit(f"--csv expects 'path:variant', got {p!r}")
        csv_path, _, variant = p.partition(":")
        entries.append({"csv": csv_path, "variant": variant})
    return wandb_io.load_run_histories(entries)


# ----- Helpers --------------------------------------------------------------

def _series(df: pd.DataFrame, variant: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df.variant == variant]
    if key not in sub.columns:
        return np.array([]), np.array([])
    s = sub[["step", key]].dropna().sort_values("step")
    return s["step"].to_numpy(), s[key].to_numpy()


def _ewma(y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0 or y.size == 0:
        return y
    return pd.Series(y).ewm(alpha=alpha, adjust=False).mean().to_numpy()


# ----- Fig. R1: raw_mse + val_loss ------------------------------------------

def plot_R1(
    df: pd.DataFrame,
    *,
    out_path: Path,
    label_overrides: dict[str, str] | None = None,
    smooth: float = 0.1,
) -> None:
    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    variants = sorted(df.variant.unique())
    imf_variants = sorted(v for v in variants if v.startswith("imf"))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)

    def _draw(ax, steps, vals, *, color, label):
        if smooth > 0 and vals.size > 1:
            ax.plot(steps, vals, color=color, alpha=0.18, linewidth=0.8)
            ax.plot(steps, _ewma(vals, smooth), color=color, label=label, linewidth=1.6)
        else:
            ax.plot(steps, vals, color=color, label=label, alpha=0.9)

    for v in variants:
        color = style.color_for(v, imf_variants)
        label = style.display_variant(v, label_overrides)
        steps, vals = _series(df, v, "raw_mse")
        if vals.size:
            _draw(axL, steps, vals, color=color, label=label)
        steps_v, vals_v = _series(df, v, "val_loss/overall")
        if not vals_v.size:
            steps_v, vals_v = _series(df, v, "val_loss")
        if vals_v.size:
            _draw(axR, steps_v, vals_v, color=color, label=label)

    # Annotate first raw_mse spike for any mf_naive curve
    for v in variants:
        if v != "mf_naive":
            continue
        steps, vals = _series(df, v, "raw_mse")
        if not vals.size:
            continue
        spike = S.first_spike_step(steps, vals)
        if spike is not None:
            axL.axvline(spike, color="#888", linestyle=":", linewidth=0.6)
            axL.annotate(
                f"first spike\n@step {spike}",
                xy=(spike, np.nanmax(vals) * 0.9),
                xytext=(8, -8),
                textcoords="offset points",
                fontsize=7,
            )

    for ax, ylabel in [(axL, "raw_mse"), (axR, "val_loss")]:
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)

    # Switch raw_mse to log y if the dynamic range > 1.5 decades
    raw_all = df.get("raw_mse")
    if raw_all is not None:
        finite = pd.to_numeric(raw_all, errors="coerce").dropna()
        if finite.size and finite.max() > 0 and finite.min() > 0:
            if np.log10(finite.max() / finite.min()) > 1.5:
                axL.set_yscale("log")

    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, 1.05))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ----- Fig. R2: head diagnostics --------------------------------------------

def plot_R2(df: pd.DataFrame, *, out_path: Path, label_overrides: dict[str, str] | None = None) -> None:
    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    variants = sorted(df.variant.unique())
    imf_variants = sorted(v for v in variants if v.startswith("imf"))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharex=True)
    titles = ["dudt_norm", "cos_u_utgt (MF) / cos_V_v (iMF)", "cos_u_v"]

    for v in variants:
        color = style.color_for(v, imf_variants)
        # Panel 1
        s, y = _series(df, v, "dudt_norm")
        if y.size:
            axes[0].plot(s, y, color=color, label=style.display_variant(v, label_overrides))
        # Panel 2
        if v.startswith("imf"):
            key = "cos_V_v"
        else:
            key = "cos_u_utgt"
        s, y = _series(df, v, key)
        if y.size:
            axes[1].plot(s, y, color=color, label=style.display_variant(v, label_overrides))
        # Panel 3
        s, y = _series(df, v, "cos_u_v")
        if y.size:
            axes[2].plot(s, y, color=color, label=style.display_variant(v, label_overrides))

    axes[1].axhline(1.0, color="#888", linestyle=":", linewidth=0.6)
    axes[0].axhline(0.0, color="#888", linestyle=":", linewidth=0.6)
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel("Training step")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, 1.06))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ----- Fig. R3: per-dataset val_loss ----------------------------------------

def plot_R3(df: pd.DataFrame, *, out_path: Path, label_overrides: dict[str, str] | None = None) -> None:
    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    variants = sorted(df.variant.unique())
    imf_variants = sorted(v for v in variants if v.startswith("imf"))

    available = {name: key for name, key in style.OXE_VAL_LOSS_KEYS.items() if key in df.columns}
    if not available:
        LOG.warning("No per-dataset val_loss columns present; skipping Fig. R3")
        return

    n = len(available)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows), sharex=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (name, key) in zip(axes_flat, available.items()):
        for v in variants:
            color = style.color_for(v, imf_variants)
            s, y = _series(df, v, key)
            if y.size:
                ax.plot(s, y, color=color, label=style.display_variant(v, label_overrides))
        ax.set_title(name)
        ax.set_xlabel("Training step")
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, 1.02))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ----- Table R1 -------------------------------------------------------------

def stability_summary(df: pd.DataFrame, *, label_overrides: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for v in sorted(df.variant.unique()):
        steps_raw, raw = _series(df, v, "raw_mse")
        steps_val, val = _series(df, v, "val_loss/overall")
        if not val.size:
            steps_val, val = _series(df, v, "val_loss")

        spike_step = S.first_spike_step(steps_raw, raw) if raw.size else None
        diverge_step = S.first_divergence_step(steps_val, val) if val.size else None

        if v.startswith("imf"):
            _, cos_final = _series(df, v, "cos_V_v")
        else:
            _, cos_final = _series(df, v, "cos_u_utgt")
        cos_tail = float(np.nanmean(cos_final[-1000:])) if cos_final.size else float("nan")

        _, dudt = _series(df, v, "dudt_norm")
        dudt_tail = float(np.nanmean(dudt[-1000:])) if dudt.size else float("nan")

        status = "diverged" if diverge_step is not None else "converged"
        rows.append({
            "variant": style.display_variant(v, label_overrides),
            "raw_mse_first_spike_step": spike_step,
            "val_loss_divergence_step": diverge_step,
            "final_cos": round(cos_tail, 3) if not np.isnan(cos_tail) else "n/a",
            "final_dudt_norm": round(dudt_tail, 1) if not np.isnan(dudt_tail) else "n/a",
            "status": status,
        })
    return pd.DataFrame(rows)


# ----- CLI ------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="JSON manifest of {csv, variant?, config?} entries.")
    g.add_argument("--csv", nargs="+", help="Inline list of `path:variant_label` pairs.")
    p.add_argument("--variant-labels", default=None, help="Path to JSON {variant: display} overrides.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--smooth",
        type=float,
        default=0.1,
        help="EWMA alpha for Fig R1 curves (default: 0.1; 0 disables smoothing).",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    label_overrides = None
    if args.variant_labels:
        with open(args.variant_labels) as f:
            label_overrides = json.load(f)

    df = load_manifest(args.manifest) if args.manifest else load_inline(args.csv)
    if "step" not in df.columns:
        raise SystemExit("Run-history CSVs must contain a 'step' or '_step' column.")

    out = Path(args.out_dir)
    plot_R1(df, out_path=out / "R1_loss_curves.pdf", label_overrides=label_overrides, smooth=args.smooth)
    plot_R2(df, out_path=out / "R2_head_diagnostics.pdf", label_overrides=label_overrides)
    plot_R3(df, out_path=out / "R3_val_loss_per_dataset.pdf", label_overrides=label_overrides)

    summary = stability_summary(df, label_overrides=label_overrides)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "R1_stability_summary.csv", index=False)
    try:
        with open(out / "R1_stability_summary.tex", "w") as f:
            f.write(summary.to_latex(index=False, escape=False))
    except Exception as e:
        LOG.warning("Could not write LaTeX: %s", e)
    LOG.info("Done. Outputs in %s", out)


if __name__ == "__main__":
    main()
