"""Pivot R5_latency.csv into the redesigned VLM + FlowTransformer layout.

The original analyzer dumps a long-form table (one row per `(variant, component)`)
with components `{vlm, dit, head, total}`. The thesis Results chapter wants a
compact format: `VLM (ms) | FlowTransformer (ms) | Total (ms)`.

FlowTransformer is reported as the **end-to-end residual** `Total - VLM` rather
than the isolated `head` measurement. The isolated `head` benchmark in
``bench_inference.py`` carries per-call dispatch and synchronisation overhead
that the end-to-end ``Total`` amortises, so `head + VLM` exceeds `Total` by a
few ms; the derived residual makes the columns sum exactly. The std for the
derived column is propagated assuming independence between the `vlm` and
`total` measurement runs: `sigma_FT = sqrt(sigma_Total^2 + sigma_VLM^2)`.

Writes `R5_latency_pivot.{csv,tex}` next to the input file.

Usage::

    python -m tools.results.latency_pivot --in docs/figures/results/R5_latency.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import style


DISPLAY_VARIANT = {"flower": "RF", "imf": "iMF"}


def fmt(mean: float, std: float) -> str:
    return f"{mean:.1f} ± {std:.1f}"


def pivot(df: pd.DataFrame, *, reference: str = "flower") -> pd.DataFrame:
    import math

    df = df.copy()
    df["variant"] = df["variant"].astype(str)

    rf_total_mean = df.query("variant == @reference and component == 'total'")["mean"].iloc[0]

    rows = []
    for variant, sub in df.groupby("variant", sort=False):
        comps = {r["component"]: (float(r["mean"]), float(r["std"])) for _, r in sub.iterrows()}
        n_samp = int(sub["n_sampling"].iloc[0])
        total_mean, total_std = comps["total"]
        vlm_mean, vlm_std = comps["vlm"]
        # FlowTransformer derived as end-to-end residual: Total - VLM. The
        # isolated `head` benchmark (comps['head']) carries per-call setup
        # overhead the end-to-end run amortises, so head + VLM > Total. Using
        # the residual makes the columns sum exactly.
        ft_mean = total_mean - vlm_mean
        ft_std = math.sqrt(total_std ** 2 + vlm_std ** 2)
        speedup = rf_total_mean / total_mean
        rows.append({
            "Variant": DISPLAY_VARIANT.get(variant, variant),
            "$N_\\mathrm{samp}$": n_samp,
            "VLM (ms)": fmt(vlm_mean, vlm_std),
            "FlowTransformer (ms)": fmt(ft_mean, ft_std),
            "Total (ms)": fmt(total_mean, total_std),
            "Speed-up": f"{speedup:.2f}\\times" + (" (ref)" if variant == reference else ""),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--reference", default="flower")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    out = pivot(df, reference=args.reference)

    csv_out = args.in_path.with_name(args.in_path.stem + "_pivot.csv")
    tex_out = args.in_path.with_name(args.in_path.stem + "_pivot.tex")

    out.to_csv(csv_out, index=False)
    out.to_latex(tex_out, index=False, escape=False)

    print(out.to_string(index=False))
    print(f"\nWrote {csv_out}")
    print(f"Wrote {tex_out}")


if __name__ == "__main__":
    main()
