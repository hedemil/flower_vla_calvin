"""LIBERO results analyzer (`docs/thesis_results.md` §4 + §6).

Inputs:
  --episodes-jsonl <glob>       per-episode outcomes (preferred, requires
                                eval-script logging patch).
  --episodes-csv <glob>         aggregate per-task SR fallback.
  --latency-json <glob>         optional benchmark JSONs for Tables R5/R7.
  --variant-labels labels.json  optional display-name overrides.
  --delta 2 5                   non-inferiority margins (pp).
  --reference-variant rf        the variant against which non-inferiority is judged.
  --out-dir thesis/figures/results/

Outputs (in --out-dir):
  R2_sr_best_epoch.{csv,tex}
  R3_sr_last_epoch.{csv,tex}
  R4_mcnemar_<suite>.{csv,tex}
  R4_sr_vs_epoch.pdf            (Fig. R4)
  R5_forest_noninf.pdf          (Fig. R5)
  R6_H1_verdict.{csv,tex}
  R7_H2_verdict.{csv,tex}
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import stats as S
from . import style
from . import wandb_io


LOG = logging.getLogger("analyze_libero_results")


# ----- I/O helpers ----------------------------------------------------------

def _expand(globs: list[str]) -> list[str]:
    out: list[str] = []
    for g in globs:
        matched = glob.glob(g)
        if not matched:
            LOG.warning("No matches for glob: %s", g)
        out.extend(sorted(matched))
    return out


def load_outcomes(args: argparse.Namespace) -> pd.DataFrame:
    """Load per-episode outcomes, preferring JSONL over aggregate CSV."""
    if args.episodes_jsonl:
        df = wandb_io.load_episodes_jsonl(_expand(args.episodes_jsonl))
        LOG.info("Loaded %d episode rows from JSONL (paired=True)", len(df))
        return df
    if args.episodes_csv:
        df = wandb_io.load_aggregate_sr_csv(_expand(args.episodes_csv), n_per_task=args.n_per_task)
        LOG.info("Loaded %d pseudo-episode rows from aggregate CSV (paired=False)", len(df))
        return df
    raise SystemExit("Must provide --episodes-jsonl or --episodes-csv")


# ----- Per-task / per-suite SR with Clopper-Pearson -------------------------

def task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, suite, epoch, task_name) with k, n, point, lo, hi."""
    rows = []
    grouped = df.groupby(["variant", "suite", "epoch", "task_name"], dropna=False)
    for (variant, suite, epoch, task_name), g in grouped:
        k = int(g["success"].sum())
        n = int(len(g))
        ci = S.clopper_pearson(k, n)
        rows.append({
            "variant": variant,
            "suite": suite,
            "epoch": int(epoch) if pd.notna(epoch) else None,
            "task_name": task_name,
            "k": k,
            "n": n,
            "sr": ci.point,
            "ci_lo": ci.lo,
            "ci_hi": ci.hi,
        })
    return pd.DataFrame(rows)


def suite_summary(task_df: pd.DataFrame) -> pd.DataFrame:
    """Average per-task CIs to a suite-level CI (per spec §2 'CI format')."""
    rows = []
    grouped = task_df.groupby(["variant", "suite", "epoch"], dropna=False)
    for (variant, suite, epoch), g in grouped:
        intervals = [S.CI(point=r.sr, lo=r.ci_lo, hi=r.ci_hi) for r in g.itertuples()]
        avg = S.average_ci(intervals)
        rows.append({
            "variant": variant,
            "suite": suite,
            "epoch": int(epoch) if pd.notna(epoch) else None,
            "n_tasks": int(len(g)),
            "sr": avg.point,
            "ci_lo": avg.lo,
            "ci_hi": avg.hi,
        })
    return pd.DataFrame(rows)


def best_epoch(suite_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the highest-SR epoch per (variant, suite)."""
    idx = suite_df.groupby(["variant", "suite"])["sr"].idxmax()
    return suite_df.loc[idx].reset_index(drop=True)


def last_epoch(suite_df: pd.DataFrame) -> pd.DataFrame:
    """Last logged epoch per (variant, suite)."""
    idx = suite_df.groupby(["variant", "suite"])["epoch"].idxmax()
    # idxmax returns NaN for all-NaN groups (single checkpoint, no epoch field);
    # fall back to best-SR row so the caller always gets valid indices.
    if idx.isna().any():
        idx = suite_df.groupby(["variant", "suite"])["sr"].idxmax()
    return suite_df.loc[idx].reset_index(drop=True)


# ----- R2 / R3 tables -------------------------------------------------------

def render_sr_table(
    suite_df: pd.DataFrame,
    *,
    variants_order: list[str],
    suites_order: list[str],
    label_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Wide table with one row per suite, one column per variant, cells =
    formatted ``point [lo, hi]`` strings.  Adds a macro-average row."""
    formatted: dict[tuple[str, str], str] = {}
    points: dict[tuple[str, str], float] = {}
    for r in suite_df.itertuples():
        formatted[(r.variant, r.suite)] = style.format_sr_ci(r.sr, r.ci_lo, r.ci_hi)
        points[(r.variant, r.suite)] = r.sr

    rows = []
    for suite in suites_order:
        row: dict[str, Any] = {"Suite": style.SUITE_DISPLAY.get(suite, suite)}
        for v in variants_order:
            row[style.display_variant(v, label_overrides)] = formatted.get((v, suite), "")
        rows.append(row)

    macro_suites = [s for s in suites_order if s != "libero_90"]
    macro_row: dict[str, Any] = {"Suite": "Macro (short-horizon)"}
    for v in variants_order:
        vals = [points[(v, s)] for s in macro_suites if (v, s) in points]
        macro_row[style.display_variant(v, label_overrides)] = (
            f"{100 * np.mean(vals):.1f}" if vals else ""
        )
    rows.append(macro_row)
    return pd.DataFrame(rows)


def diff_flags(best: pd.DataFrame, last: pd.DataFrame, *, threshold_pp: float = 5.0) -> pd.DataFrame:
    """Mark cells where best- and last-epoch SR differ by more than threshold (pp)."""
    merged = best.merge(
        last,
        on=["variant", "suite"],
        suffixes=("_best", "_last"),
    )
    merged["delta_pp"] = 100 * (merged["sr_best"] - merged["sr_last"])
    merged["flag"] = merged["delta_pp"].abs() > threshold_pp
    return merged[["variant", "suite", "delta_pp", "flag"]]


# ----- R4 McNemar -----------------------------------------------------------

def mcnemar_per_suite(
    df: pd.DataFrame,
    suite_df_best: pd.DataFrame,
    *,
    variants_order: list[str],
) -> dict[str, pd.DataFrame]:
    """For each suite, build the lower-triangle McNemar p-value matrix.

    Pairs episodes by ``(task_name, episode_index)`` at the best-of-epoch
    chosen for each (variant, suite).  Refuses to run on unpaired data.
    """
    paired = df.attrs.get("paired", True)
    out: dict[str, pd.DataFrame] = {}
    suites = sorted(df["suite"].dropna().unique())
    for suite in suites:
        mat = pd.DataFrame(
            index=variants_order, columns=variants_order, dtype=object
        )
        for v in variants_order:
            mat.loc[v, v] = "—"
        if not paired:
            for a, b in combinations(variants_order, 2):
                mat.loc[a, b] = "n/a (unpaired)"
                mat.loc[b, a] = ""
            out[suite] = mat
            continue

        # Get best epoch per variant for this suite
        best_epochs = {
            r.variant: r.epoch
            for r in suite_df_best[suite_df_best.suite == suite].itertuples()
        }
        for a, b in combinations(variants_order, 2):
            if a not in best_epochs or b not in best_epochs:
                mat.loc[a, b] = "—"
                continue
            ea = best_epochs[a]
            eb = best_epochs[b]
            epoch_mask_a = df["epoch"].isna() if ea is None else (df["epoch"] == ea)
            epoch_mask_b = df["epoch"].isna() if eb is None else (df["epoch"] == eb)
            sub_a = df[(df.variant == a) & (df.suite == suite) & epoch_mask_a]
            sub_b = df[(df.variant == b) & (df.suite == suite) & epoch_mask_b]
            merged = sub_a.merge(
                sub_b, on=["task_name", "episode_index"], suffixes=("_a", "_b")
            )
            if merged.empty:
                mat.loc[a, b] = "—"
                continue
            n_pairs = max(1, len(list(combinations(variants_order, 2))))
            p = S.paired_mcnemar(merged["success_a"].tolist(), merged["success_b"].tolist())
            marker = style.significance_marker(p, n_corrections=n_pairs)
            mat.loc[a, b] = f"{style.format_pvalue(p)}{marker}"
            mat.loc[b, a] = ""  # lower-triangle only
        out[suite] = mat
    return out


# ----- R5 forest plot data --------------------------------------------------

def paired_forest_rows(
    df: pd.DataFrame,
    suite_df_best: pd.DataFrame,
    *,
    reference: str,
    variants_order: list[str],
) -> pd.DataFrame:
    """One row per (variant, suite) (excluding reference) with paired diff CI."""
    paired = df.attrs.get("paired", True)
    rows = []
    for v in variants_order:
        if v == reference:
            continue
        for suite in sorted(df["suite"].dropna().unique()):
            row = {"variant": v, "suite": suite, "diff": np.nan, "lo": np.nan, "hi": np.nan, "paired": paired}
            if not paired:
                rows.append(row)
                continue
            ev = suite_df_best.loc[
                (suite_df_best.variant == v) & (suite_df_best.suite == suite), "epoch"
            ]
            er = suite_df_best.loc[
                (suite_df_best.variant == reference) & (suite_df_best.suite == suite), "epoch"
            ]
            if ev.empty or er.empty:
                rows.append(row)
                continue
            ev = ev.iloc[0]
            er = er.iloc[0]
            epoch_mask_v = df["epoch"].isna() if ev is None else (df["epoch"] == ev)
            epoch_mask_r = df["epoch"].isna() if er is None else (df["epoch"] == er)
            sub_v = df[(df.variant == v) & (df.suite == suite) & epoch_mask_v]
            sub_r = df[(df.variant == reference) & (df.suite == suite) & epoch_mask_r]
            merged = sub_v.merge(sub_r, on=["task_name", "episode_index"], suffixes=("_v", "_r"))
            if merged.empty:
                rows.append(row)
                continue
            ci = S.paired_diff_ci(merged["success_v"].tolist(), merged["success_r"].tolist())
            row.update({"diff": ci.point, "lo": ci.lo, "hi": ci.hi})
            rows.append(row)
    return pd.DataFrame(rows)


# ----- R6 / R7 verdicts -----------------------------------------------------

def h1_verdict(forest: pd.DataFrame, deltas_pp: list[float]) -> pd.DataFrame:
    rows = []
    for r in forest.itertuples():
        row = {"variant": r.variant, "suite": r.suite}
        for d in deltas_pp:
            verdict = (
                S.non_inferiority_verdict(r.diff, r.lo, d / 100.0)
                if r.paired and not (np.isnan(r.diff) or np.isnan(r.lo))
                else "n/a"
            )
            row[f"δ={d}pp"] = verdict
        rows.append(row)
    return pd.DataFrame(rows)


def h2_verdict(latency_summary: pd.DataFrame, *, reference: str) -> pd.DataFrame:
    """Per Table R7: pass if (mean+2σ) of variant total latency < RF mean."""
    total = latency_summary[latency_summary.component == "total"]
    if total.empty:
        return pd.DataFrame()
    ref_row = total[total.variant == reference]
    if ref_row.empty:
        LOG.warning("Reference variant %s not in latency data; skipping H2 verdict", reference)
        return pd.DataFrame()
    rf_mean = float(ref_row.iloc[0]["mean"])
    rf_std = float(ref_row.iloc[0]["std"])
    rows = []
    for r in total.itertuples():
        if r.variant == reference:
            continue
        verdict = S.latency_speedup_verdict(rf_mean, rf_std, float(r.mean), float(r.std))
        rows.append({
            "variant": r.variant,
            "n_sampling": int(r.n_sampling),
            "mean_ms": float(r.mean),
            "std_ms": float(r.std),
            "speedup_x": rf_mean / float(r.mean) if r.mean else float("inf"),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


# ----- Plotting -------------------------------------------------------------

def plot_sr_vs_epoch(
    suite_df: pd.DataFrame,
    suite_df_best: pd.DataFrame,
    *,
    out_path: Path,
    variants_order: list[str],
    label_overrides: dict[str, str] | None = None,
) -> None:
    """Fig. R4: five panels (one per suite), x=epoch, y=suite-averaged SR,
    shaded band = average of per-task CIs, vertical tick at the chosen best."""
    import matplotlib.pyplot as plt

    style.apply_thesis_rcparams()
    suites = [s for s in style.SUITE_DISPLAY if s in suite_df["suite"].unique()]
    if not suites:
        LOG.warning("No known suites in data; skipping Fig. R4")
        return
    n_panels = len(suites)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 2.4), sharey=True)
    if n_panels == 1:
        axes = [axes]
    imf_variants = sorted(v for v in variants_order if v.startswith("imf"))
    for ax, suite in zip(axes, suites):
        for v in variants_order:
            sub = suite_df[(suite_df.variant == v) & (suite_df.suite == suite)].sort_values("epoch")
            if sub.empty:
                continue
            color = style.color_for(v, imf_variants)
            # Use sequential index when epoch is not recorded (single checkpoint).
            x = sub["epoch"].fillna(pd.Series(range(len(sub)), index=sub.index))
            ax.plot(x, 100 * sub.sr, color=color, label=style.display_variant(v, label_overrides))
            ax.fill_between(x, 100 * sub.ci_lo, 100 * sub.ci_hi, color=color, alpha=0.15)
            best = suite_df_best[(suite_df_best.variant == v) & (suite_df_best.suite == suite)]
            if not best.empty:
                ep = best.iloc[0]["epoch"]
                if pd.notna(ep):
                    ax.axvline(ep, color=color, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_title(style.SUITE_DISPLAY.get(suite, suite))
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Success rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, 1.02))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def plot_forest_noninf(
    forest: pd.DataFrame,
    *,
    out_path: Path,
    deltas_pp: list[float],
    label_overrides: dict[str, str] | None = None,
) -> None:
    """Fig. R5: forest plot of paired SR differences vs. the reference."""
    import matplotlib.pyplot as plt

    if forest.empty or not forest["paired"].any():
        LOG.warning("No paired data for Fig. R5; skipping")
        return
    style.apply_thesis_rcparams()
    rows = forest.dropna(subset=["diff"]).reset_index(drop=True)
    if rows.empty:
        LOG.warning("Forest rows are empty after NaN drop; skipping Fig. R5")
        return
    n_panels = len(deltas_pp)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, max(2.5, 0.3 * len(rows))), sharey=True)
    if n_panels == 1:
        axes = [axes]
    rows = rows.assign(label=lambda d: d.variant.map(lambda v: style.display_variant(v, label_overrides)) + " · " + d.suite.map(lambda s: style.SUITE_DISPLAY.get(s, s)))
    y = np.arange(len(rows))
    for ax, delta in zip(axes, deltas_pp):
        ax.errorbar(
            100 * rows["diff"],
            y,
            xerr=[100 * (rows["diff"] - rows["lo"]), 100 * (rows["hi"] - rows["diff"])],
            fmt="o",
            color="#222222",
            ecolor="#888888",
            capsize=2,
            markersize=3,
        )
        ax.axvline(0, color="#444444", linewidth=0.6)
        ax.axvline(-delta, color="#D55E00", linewidth=0.8, linestyle="--", label=f"-δ = -{delta} pp")
        ax.set_yticks(y)
        ax.set_yticklabels(rows["label"])
        ax.invert_yaxis()
        ax.set_xlabel("SR difference vs. reference (pp)")
        ax.set_title(f"δ = {delta} pp")
        ax.legend(loc="lower right", frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


def plot_pareto(
    suite_df_best: pd.DataFrame,
    latency_summary: pd.DataFrame,
    *,
    out_path: Path,
    variants_order: list[str],
    label_overrides: dict[str, str] | None = None,
) -> None:
    """Fig. R6: latency-vs-SR scatter (Pareto plot).

    x = total latency mean (ms), y = best-of-epoch suite SR (%). One marker
    per (variant, suite); shape encodes suite, colour encodes variant. Dashed
    line through Pareto-dominating points.
    """
    import matplotlib.pyplot as plt

    total = latency_summary[latency_summary.component == "total"]
    if total.empty or suite_df_best.empty:
        LOG.warning("No latency or SR data; skipping Fig. R6")
        return
    style.apply_thesis_rcparams()
    suites_present = sorted(suite_df_best["suite"].unique())
    suite_markers = ["o", "s", "^", "D", "P", "X"]
    marker_map = {s: suite_markers[i % len(suite_markers)] for i, s in enumerate(suites_present)}

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    imf_variants = sorted(v for v in variants_order if v.startswith("imf"))
    points = []
    seen_variants: set[str] = set()
    seen_suites: set[str] = set()
    for v in variants_order:
        lat_row = total[total.variant == v]
        if lat_row.empty:
            continue
        lat_ms = float(lat_row.iloc[0]["mean"])
        color = style.color_for(v, imf_variants)
        for suite in suites_present:
            sr_row = suite_df_best[(suite_df_best.variant == v) & (suite_df_best.suite == suite)]
            if sr_row.empty:
                continue
            sr_pct = 100 * float(sr_row.iloc[0]["sr"])
            ax.scatter(
                lat_ms, sr_pct,
                marker=marker_map[suite], color=color, s=42,
                edgecolors="black", linewidths=0.3,
            )
            if v not in seen_variants:
                ax.scatter([], [], marker="o", color=color,
                           label=style.display_variant(v, label_overrides))
                seen_variants.add(v)
            if suite not in seen_suites:
                ax.scatter([], [], marker=marker_map[suite], color="#888888",
                           label=style.SUITE_DISPLAY.get(suite, suite))
                seen_suites.add(suite)
            points.append((lat_ms, sr_pct))

    if points:
        points_sorted = sorted(points, key=lambda p: (p[0], -p[1]))
        frontier = []
        best_y = -1.0
        for x, y in points_sorted:
            if y > best_y:
                frontier.append((x, y))
                best_y = y
        if len(frontier) >= 2:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, linestyle="--", color="#444444", linewidth=0.8,
                    label="Pareto frontier")

    ax.set_xlabel("Total inference latency (ms)")
    ax.set_ylabel("Best-of-epoch SR (%)")
    ax.legend(loc="best", frameon=False, fontsize=7, ncol=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ----- Output writers -------------------------------------------------------

def _write(df: pd.DataFrame, out_dir: Path, stem: str, *, index: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=index)
    try:
        with open(out_dir / f"{stem}.tex", "w") as f:
            f.write(df.to_latex(index=index, escape=False))
    except Exception as e:  # to_latex requires Jinja2 in some pandas versions
        LOG.warning("Could not write LaTeX for %s: %s", stem, e)


# ----- CLI ------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes-jsonl", nargs="*", default=[], help="Glob(s) for per-episode JSONL files.")
    p.add_argument("--episodes-csv", nargs="*", default=[], help="Glob(s) for aggregate per-task SR CSVs (fallback).")
    p.add_argument("--latency-json", nargs="*", default=[], help="Glob(s) for benchmark JSONs (Table R5/R7).")
    p.add_argument("--variant-labels", default=None, help="Path to JSON `{variant: display}` overrides.")
    p.add_argument("--reference-variant", default="rf", help="Variant used as the non-inferiority reference.")
    p.add_argument("--delta", nargs="+", type=float, default=[2.0, 5.0], help="Non-inferiority margins in pp.")
    p.add_argument("--n-per-task", type=int, default=20, help="Episodes per task (only for aggregate-CSV fallback).")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    label_overrides = None
    if args.variant_labels:
        with open(args.variant_labels) as f:
            label_overrides = json.load(f)

    df = load_outcomes(args)
    variants_order = sorted(df["variant"].dropna().unique().tolist())
    suites_order = [s for s in style.SUITE_DISPLAY if s in df["suite"].unique()] or sorted(df["suite"].dropna().unique())

    task_df = task_summary(df)
    suite_df = suite_summary(task_df)
    best = best_epoch(suite_df)
    last = last_epoch(suite_df)

    out = Path(args.out_dir)

    # --- R2 / R3 ----------------------------------------------------------
    r2 = render_sr_table(best, variants_order=variants_order, suites_order=suites_order, label_overrides=label_overrides)
    r3 = render_sr_table(last, variants_order=variants_order, suites_order=suites_order, label_overrides=label_overrides)
    _write(r2, out, "R2_sr_best_epoch")
    _write(r3, out, "R3_sr_last_epoch")
    flags = diff_flags(best, last)
    _write(flags, out, "R3_best_vs_last_flags")

    # --- R4 ---------------------------------------------------------------
    matrices = mcnemar_per_suite(df, best, variants_order=variants_order)
    for suite, mat in matrices.items():
        _write(mat.reset_index().rename(columns={"index": "variant"}), out, f"R4_mcnemar_{suite}")

    # --- Fig R4 -----------------------------------------------------------
    plot_sr_vs_epoch(
        suite_df,
        best,
        out_path=out / "R4_sr_vs_epoch.pdf",
        variants_order=variants_order,
        label_overrides=label_overrides,
    )

    # --- Fig R5 + Table R6 ------------------------------------------------
    forest = paired_forest_rows(df, best, reference=args.reference_variant, variants_order=variants_order)
    plot_forest_noninf(forest, out_path=out / "R5_forest_noninf.pdf", deltas_pp=args.delta, label_overrides=label_overrides)
    r6 = h1_verdict(forest, args.delta)
    _write(r6, out, "R6_H1_verdict")

    # --- Latency (R5 + R7 + Fig R6) --------------------------------------
    if args.latency_json:
        lat_df = wandb_io.load_latency_jsons(_expand(args.latency_json))
        if not lat_df.empty:
            lat_summary = wandb_io.latency_summary(lat_df)
            _write(lat_summary, out, "R5_latency")
            plot_pareto(
                best,
                lat_summary,
                out_path=out / "R6_pareto.pdf",
                variants_order=variants_order,
                label_overrides=label_overrides,
            )
            r7 = h2_verdict(lat_summary, reference=args.reference_variant)
            if not r7.empty:
                _write(r7, out, "R7_H2_verdict")

    LOG.info("Done. Outputs in %s", out)


if __name__ == "__main__":
    main()
