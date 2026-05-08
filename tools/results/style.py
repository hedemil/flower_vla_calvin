"""Reporting conventions (`docs/thesis_results.md` §2).

Centralizes the Okabe-Ito palette, display labels, matplotlib rcParams,
and number formatters so every figure/table renders identically.
"""

from __future__ import annotations

from typing import Mapping


OKABE_ITO_BLUE = "#0072B2"
OKABE_ITO_VERMILION = "#D55E00"
OKABE_ITO_GREEN = "#009E73"
OKABE_ITO_GREEN_LIGHT = "#4FC3A1"
OKABE_ITO_GREEN_DARK = "#00785A"
OKABE_ITO_GREEN_DARKEST = "#005A43"


# Canonical variant -> color.  Off-default ablations get green shades; any
# variant key not listed here cycles through the green shades in order so
# discover_variants() output never collides with rf/mf colors.
VARIANT_COLOR: dict[str, str] = {
    "rf": OKABE_ITO_BLUE,
    "flower": OKABE_ITO_BLUE,
    "mf_naive": OKABE_ITO_VERMILION,
    "imf": OKABE_ITO_GREEN,
}

GREEN_SHADES = [
    OKABE_ITO_GREEN,
    OKABE_ITO_GREEN_LIGHT,
    OKABE_ITO_GREEN_DARK,
    OKABE_ITO_GREEN_DARKEST,
]


def color_for(variant: str, observed_imf_variants: list[str] | None = None) -> str:
    """Return the plot color for a variant key.

    `rf` and `mf_naive` are fixed.  `imf` defaults to green.  Other iMF
    ablations (any key starting with `imf_`) get distinct green shades by
    their position in `observed_imf_variants` (reproducible ordering).
    """
    if variant in VARIANT_COLOR:
        return VARIANT_COLOR[variant]
    if variant.startswith("imf"):
        ordered = observed_imf_variants or []
        if variant in ordered:
            return GREEN_SHADES[ordered.index(variant) % len(GREEN_SHADES)]
        return OKABE_ITO_GREEN
    # Unknown family: fall back to a neutral grey
    return "#666666"


VARIANT_DISPLAY: dict[str, str] = {
    "rf": "RF",
    "flower": "RF",
    "mf_naive": "MF (naïve)",
    "imf": "iMF",
}


def display_variant(variant: str, overrides: Mapping[str, str] | None = None) -> str:
    """Pretty-print a variant key. `imf_r0.50_lh10` -> `iMF, ρ=0.50, L_head=10`."""
    if overrides and variant in overrides:
        return overrides[variant]
    if variant in VARIANT_DISPLAY:
        return VARIANT_DISPLAY[variant]
    if variant.startswith("imf_"):
        parts = []
        for token in variant.split("_")[1:]:
            if token.startswith("r"):
                try:
                    parts.append(f"ρ={float(token[1:]):.2f}")
                except ValueError:
                    parts.append(token)
            elif token.startswith("lh"):
                parts.append(f"$L_\\text{{head}}$={token[2:]}")
            else:
                parts.append(token)
        return "iMF, " + ", ".join(parts)
    return variant


SUITE_DISPLAY: dict[str, str] = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long-10",
    "libero_90": "Short-90",
}

SUITE_TASKS: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

# Per-dataset W&B keys for Open-X val-loss (Fig. R3).  Verbatim from the
# pretraining trainer (DatasetMetricsTracker).
OXE_VAL_LOSS_KEYS: dict[str, str] = {
    "fractal20220817_data": "val_loss/fractal20220817_data",
    "bridge_dataset": "val_loss/bridge",
    "eef_droid": "val_loss/droid",
    "dobbe": "val_loss/dobbe",
    "bc_z": "val_loss/bc_z:0.1.0",
    "cmu_play_fusion": "val_loss/cmu_play_fusion",
    "stanford_hydra_dataset": "val_loss/stanford_hydra_dataset_converted_externally_to_rlds",
    "libero_10_no_noops": "val_loss/libero_10_no_noops",
    "libero_goal_no_noops": "val_loss/libero_goal_no_noops",
}


def apply_thesis_rcparams() -> None:
    """Set matplotlib rcParams for thesis-quality vector output."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def format_sr_ci(point: float, lo: float, hi: float) -> str:
    """`45.0 [38.1, 52.1]` (pp, 0.1 decimals).  Inputs are fractions in [0, 1]."""
    return f"{100 * point:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def format_latency(mean_ms: float, std_ms: float) -> str:
    """`12.3 ± 0.4 ms` (0.1 ms decimals)."""
    return f"{mean_ms:.1f} ± {std_ms:.1f} ms"


def format_pvalue(p: float) -> str:
    """3 significant figures, e.g. `0.00210`."""
    if p == 0:
        return "0.000"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3g}"


def significance_marker(p: float, n_corrections: int = 1) -> str:
    """`*`/`**`/`***` after Bonferroni correction (`thesis_results.md` §2)."""
    p_eff = p * n_corrections
    if p_eff < 0.001:
        return "***"
    if p_eff < 0.01:
        return "**"
    if p_eff < 0.05:
        return "*"
    return ""
