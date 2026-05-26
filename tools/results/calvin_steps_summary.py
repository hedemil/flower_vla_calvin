"""Average env-steps to finish a CALVIN subtask, per model and per NFE.

Reads one or more ``rollout_episodes.jsonl`` files (written by
``flower/evaluation/flower_evaluate.py`` / the RolloutLongHorizon callback) and
reports, over *successful* subtask attempts only, the mean / median / std of the
``steps`` field — aggregated and per subtask name. For pairs of runs on the same
benchmark it also reports the paired mean Δsteps (Wilcoxon signed-rank) on
subtasks both runs solved, pairing by ``(sequence_index, subtask_position)``.

Each run is labelled ``<model>_<bench>[_nfe<N>]`` so the label encodes model
(``imf``/``flower``), benchmark (``d``/``dd``/``abcd``) and NFE. iMF defaults to
NFE=1. Same seed + same ``num_sequences`` => identical eval chains across runs of
one benchmark, so ``sequence_index`` alignment is exact.

Steps semantics (see flower_evaluate.rollout): on success ``steps`` is the env
step that solved the subtask (1..ep_len); on failure it is capped at ep_len=360;
un-attempted subtasks after a broken chain are ``-1``. We keep only
``success==1 and steps>=1``.

Usage::

    python -m tools.results.calvin_steps_summary \\
        --run imf_dd=tools/results/_cache/calvin/imf_dd/rollout_episodes.jsonl \\
        --run flower_dd_nfe1=.../flower_dd_nfe1/rollout_episodes.jsonl \\
        --run flower_dd_nfe2=.../flower_dd_nfe2/rollout_episodes.jsonl \\
        --run flower_dd_nfe3=.../flower_dd_nfe3/rollout_episodes.jsonl \\
        --run flower_dd_nfe4=.../flower_dd_nfe4/rollout_episodes.jsonl \\
        --out-dir docs/figures/results/
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scstats

from . import style


LABEL_RE = re.compile(r"^(?P<model>imf|flower)_(?P<bench>abcd|dd|d)(?:_nfe(?P<nfe>\d+))?$")
BENCH_DISPLAY = {"d": "D→D", "dd": "D→D", "abcd": "ABC→D"}


def parse_label(label: str) -> dict:
    """Split a run label into (model, bench, nfe). Tolerant: unknown -> raw."""
    m = LABEL_RE.match(label)
    if not m:
        return {"label": label, "model": None, "bench": None, "nfe": None}
    model = m.group("model")
    nfe = m.group("nfe")
    if nfe is not None:
        nfe = int(nfe)
    elif model == "imf":
        nfe = 1  # iMF eval uses num_sampling_steps=1
    return {"label": label, "model": model, "bench": m.group("bench"), "nfe": nfe}


def load_subtasks(path: Path) -> pd.DataFrame:
    """Load kind=subtask records. If several epochs exist, keep the latest."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") != "subtask":
                continue
            rows.append(rec)
    if not rows:
        raise SystemExit(f"No subtask records in {path}")
    df = pd.DataFrame(rows)
    if "epoch" in df.columns and df["epoch"].nunique() > 1:
        df = df[df["epoch"] == df["epoch"].max()].copy()
    return df.reset_index(drop=True)


def _summary_row(meta: dict, df: pd.DataFrame) -> dict:
    attempted = df[df["steps"] >= 0]
    succ = df[(df["success"] == 1) & (df["steps"] >= 1)]["steps"]
    n_succ = int(len(succ))
    return {
        "label": meta["label"],
        "model": meta["model"],
        "bench": meta["bench"],
        "nfe": meta["nfe"],
        "n_attempted": int(len(attempted)),
        "n_success": n_succ,
        "mean_steps": float(succ.mean()) if n_succ else float("nan"),
        "std_steps": float(succ.std(ddof=1)) if n_succ > 1 else (0.0 if n_succ == 1 else float("nan")),
        "median_steps": float(succ.median()) if n_succ else float("nan"),
    }


def _paired_delta(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[float, float, int]:
    """Paired Δsteps (a - b) on subtasks both runs solved. Returns (mean Δ, p, n)."""
    a = df_a[(df_a["success"] == 1) & (df_a["steps"] >= 1)]
    b = df_b[(df_b["success"] == 1) & (df_b["steps"] >= 1)]
    merged = a.merge(b, on=["sequence_index", "subtask_position"], suffixes=("_a", "_b"))
    if merged.empty:
        return float("nan"), float("nan"), 0
    delta = merged["steps_a"] - merged["steps_b"]
    if len(delta) < 2 or (delta == 0).all():
        return float(delta.mean()), float("nan"), int(len(delta))
    res = scstats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return float(delta.mean()), float(res.pvalue), int(len(delta))


def _pick_reference(bench_rows: list[dict]) -> str | None:
    """Reference for paired Δ within a benchmark: the iMF run if present,
    else the lowest-NFE flower run."""
    imf = [r for r in bench_rows if r["model"] == "imf"]
    if imf:
        return imf[0]["label"]
    flower = sorted((r for r in bench_rows if r["nfe"] is not None), key=lambda r: r["nfe"])
    return flower[0]["label"] if flower else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                    help="Run label and its rollout_episodes.jsonl. Repeatable.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-plot", action="store_true", help="Skip the PDF figure.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        meta = parse_label(label)
        runs[label] = meta
        frames[label] = load_subtasks(Path(path))

    # ---- aggregate summary -------------------------------------------------
    summary = pd.DataFrame([_summary_row(runs[l], frames[l]) for l in runs])

    # paired Δsteps vs the per-benchmark reference
    summary["delta_vs_ref"] = float("nan")
    summary["wilcoxon_p"] = float("nan")
    summary["n_paired"] = 0
    summary["reference"] = ""
    for bench, grp in summary.groupby("bench", dropna=False):
        ref = _pick_reference(grp.to_dict("records"))
        if ref is None:
            continue
        for label in grp["label"]:
            mask = summary["label"] == label
            summary.loc[mask, "reference"] = ref
            if label == ref:
                summary.loc[mask, "delta_vs_ref"] = 0.0
                summary.loc[mask, "n_paired"] = int(
                    ((frames[ref]["success"] == 1) & (frames[ref]["steps"] >= 1)).sum())
                continue
            d, p, n = _paired_delta(frames[label], frames[ref])
            summary.loc[mask, "delta_vs_ref"] = d
            summary.loc[mask, "wilcoxon_p"] = p
            summary.loc[mask, "n_paired"] = n

    summary = summary.sort_values(
        ["bench", "model", "nfe"], na_position="last").reset_index(drop=True)

    csv_path = args.out_dir / "R_calvin_steps.csv"
    summary.to_csv(csv_path, index=False)

    # ---- per-task table ----------------------------------------------------
    per_task_rows = []
    for label, df in frames.items():
        succ = df[(df["success"] == 1) & (df["steps"] >= 1)]
        g = succ.groupby("subtask_name")["steps"].agg(["count", "mean", "median", "std"])
        for name, row in g.iterrows():
            per_task_rows.append({
                "label": label, "subtask_name": name,
                "n_success": int(row["count"]), "mean_steps": float(row["mean"]),
                "median_steps": float(row["median"]),
                "std_steps": float(row["std"]) if not math.isnan(row["std"]) else 0.0,
            })
    per_task = pd.DataFrame(per_task_rows)
    per_task_csv = args.out_dir / "R_calvin_steps_pertask.csv"
    per_task.to_csv(per_task_csv, index=False)
    # wide pivot of mean steps for quick scanning
    if not per_task.empty:
        pivot = per_task.pivot(index="subtask_name", columns="label", values="mean_steps")
        pivot.to_csv(args.out_dir / "R_calvin_steps_pertask_wide.csv")

    # ---- LaTeX -------------------------------------------------------------
    tex_path = args.out_dir / "R_calvin_steps.tex"
    lines = [
        "\\begin{tabular}{llcccrc}",
        "\\toprule",
        "Benchmark & Model & NFE & $n_\\mathrm{succ}$ & Mean steps ($\\pm\\sigma$) & "
        "$\\overline{\\Delta}$ vs ref & Wilcoxon $p$ \\\\",
        "\\midrule",
    ]
    for r in summary.itertuples():
        bench = BENCH_DISPLAY.get(r.bench, r.bench if r.bench else "--")
        model = style.VARIANT_DISPLAY.get(r.model, r.model if r.model else r.label)
        nfe = str(int(r.nfe)) if pd.notna(r.nfe) else "--"
        if math.isnan(r.mean_steps):
            mean_fmt = "[no data]"
        else:
            mean_fmt = f"${r.mean_steps:.1f} \\pm {r.std_steps:.1f}$"
        if r.label == r.reference or math.isnan(r.delta_vs_ref):
            d_fmt, p_fmt = "--", "--"
        else:
            d_fmt = f"${r.delta_vs_ref:+.1f}$"
            p_fmt = style.format_pvalue(r.wilcoxon_p) if not math.isnan(r.wilcoxon_p) else "--"
        lines.append(
            f"{bench} & {model} & {nfe} & {r.n_success} & {mean_fmt} & {d_fmt} & {p_fmt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    tex_path.write_text("\n".join(lines))

    # ---- figure: mean steps vs NFE ----------------------------------------
    if not args.no_plot:
        import matplotlib.pyplot as plt

        style.apply_thesis_rcparams()
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for bench, grp in summary.groupby("bench", dropna=False):
            bench_lbl = BENCH_DISPLAY.get(bench, str(bench))
            fl = grp[(grp["model"] == "flower") & grp["nfe"].notna()].sort_values("nfe")
            if not fl.empty:
                ax.errorbar(fl["nfe"], fl["mean_steps"],
                            yerr=fl["std_steps"] / np.sqrt(fl["n_success"].clip(lower=1)),
                            marker="o", capsize=2, label=f"RF {bench_lbl}",
                            color=style.color_for("rf"))
            im = grp[grp["model"] == "imf"]
            for _, ir in im.iterrows():
                ax.scatter([ir["nfe"]], [ir["mean_steps"]], marker="*", s=120, zorder=5,
                           color=style.color_for("imf"), label=f"iMF {bench_lbl}")
        ax.set_xlabel("NFE (number of function evaluations)")
        ax.set_ylabel("Mean env steps to solve a subtask")
        ax.set_title("Steps to completion vs NFE (successful subtasks)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig_path = args.out_dir / "R_calvin_steps.pdf"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Wrote {fig_path}")

    print(f"Wrote {csv_path}")
    print(f"Wrote {per_task_csv}")
    print(f"Wrote {tex_path}")
    print()
    cols = ["bench", "model", "nfe", "n_success", "mean_steps", "std_steps",
            "median_steps", "delta_vs_ref", "wilcoxon_p", "n_paired"]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
