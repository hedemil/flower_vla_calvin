"""Loaders for W&B exports and patched-eval outputs.

Two input shapes are supported:

* **Run-history CSVs** (Fig. R1/R2/R3 source).  W&B "Export → CSV" gives one
  CSV per run, with columns for every logged scalar key (one per step or
  per epoch).  We accept multiple CSVs and concatenate after attaching a
  ``variant`` column derived from the run config.

* **Episode JSONLs** (Tables R2-R7 source).  Produced by the patched
  ``flower_eval_libero.py`` (see Gap 1 in the plan).  Each line is a JSON
  object with ``{task_name, episode_index, success, suite, variant, epoch}``.

A fallback CSV format (one row per task with aggregate SR) is also handled
for backwards compatibility with runs that pre-date the logging patch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


# ----- Run history (training curves) ----------------------------------------

def _normalize_step_column(df: pd.DataFrame) -> pd.DataFrame:
    """W&B uses '_step' for the global step counter; normalize to 'step'."""
    if "step" not in df.columns and "_step" in df.columns:
        df = df.rename(columns={"_step": "step"})
    if "step" not in df.columns and "global_step" in df.columns:
        df = df.rename(columns={"global_step": "step"})
    return df


def load_run_history(
    csv_path: str | Path,
    *,
    variant: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Load one W&B run-history CSV.

    Either ``variant`` or ``config`` must be provided so each row can be
    tagged.  If ``config`` is given, ``variants.identify_variant`` is used to
    derive the variant label.
    """
    df = pd.read_csv(csv_path)
    df = _normalize_step_column(df)
    if variant is None:
        if config is None:
            raise ValueError("Provide either `variant` or `config` to label the run.")
        from .variants import identify_variant
        variant = identify_variant(config).label()
    df["variant"] = variant
    df["source"] = str(csv_path)
    return df


def load_run_histories(
    sources: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Load and concatenate several run-history CSVs.

    Each entry in ``sources`` is a dict with at least ``csv`` and one of
    ``variant`` or ``config`` (matching :func:`load_run_history`).
    """
    frames = []
    for entry in sources:
        df = load_run_history(
            entry["csv"],
            variant=entry.get("variant"),
            config=entry.get("config"),
        )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_run_history_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load run histories from a JSON manifest.

    Manifest schema:

    ```json
    [
      {"csv": "exports/rf.csv", "variant": "rf"},
      {"csv": "exports/imf.csv", "config": {"model": {"_target_": "MeanFlowerVLA", "use_imf": true, "ratio": 0.25, "imf_head_depth": 8}}},
      ...
    ]
    ```

    Relative `csv` paths are resolved against the manifest's parent
    directory, so a manifest at `wandb_exports/manifest.json` referencing
    `rf.csv` finds `wandb_exports/rf.csv`.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        entries = json.load(f)
    base = manifest_path.parent
    for entry in entries:
        csv = Path(entry["csv"])
        if not csv.is_absolute():
            entry["csv"] = str(base / csv)
    return load_run_histories(entries)


# ----- LIBERO episode outcomes ----------------------------------------------

EPISODES_SCHEMA = ["variant", "suite", "epoch", "task_name", "episode_index", "success", "steps"]


def load_episodes_jsonl(
    paths: Iterable[str | Path],
    *,
    default_variant: str | None = None,
    default_suite: str | None = None,
    default_epoch: int | None = None,
) -> pd.DataFrame:
    """Load per-episode outcomes (preferred path; requires the eval patch).

    Each JSONL line should have at least ``task_name``, ``episode_index``,
    ``success``.  ``variant``, ``suite``, ``epoch`` may be present per-line
    or supplied as defaults (when one JSONL file == one (variant, suite,
    epoch) cell, the patched eval writes them as defaults).

    Returns a DataFrame with the schema in :data:`EPISODES_SCHEMA`,
    ``success`` cast to ``int`` (0/1), tagged with ``paired=True`` attribute.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows.append({
                    "variant": rec.get("variant", default_variant),
                    "suite": rec.get("suite", default_suite),
                    "epoch": rec.get("epoch", default_epoch),
                    "task_name": rec["task_name"],
                    "episode_index": int(rec["episode_index"]),
                    "success": int(bool(rec["success"])),
                    "steps": int(rec["steps"]) if "steps" in rec else None,
                })
    df = pd.DataFrame(rows, columns=EPISODES_SCHEMA)
    df.attrs["paired"] = True
    if df["variant"].isna().any() or df["suite"].isna().any():
        missing = df[df[["variant", "suite"]].isna().any(axis=1)]
        raise ValueError(
            f"{len(missing)} episodes are missing variant/suite tagging "
            "(supply default_variant/default_suite, or include them per-line)."
        )
    return df


def load_aggregate_sr_csv(
    paths: Iterable[str | Path],
    *,
    n_per_task: int = 20,
) -> pd.DataFrame:
    """Fallback for runs that pre-date the eval patch.

    Each CSV row gives ``(variant, suite, epoch, task_name, success_rate)``.
    Expanded into pseudo-episode rows with the right number of successes.
    Episodes are pseudo-indexed 0..n-1 deterministically; **pairing across
    variants is meaningless** under this loader, so the resulting DataFrame
    is tagged with ``paired=False`` and downstream stats helpers should
    refuse McNemar against it.
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        required = {"variant", "suite", "epoch", "task_name", "success_rate"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        rows = []
        for _, r in df.iterrows():
            n_succ = int(round(float(r["success_rate"]) * n_per_task))
            for i in range(n_per_task):
                rows.append({
                    "variant": r["variant"],
                    "suite": r["suite"],
                    "epoch": int(r["epoch"]),
                    "task_name": r["task_name"],
                    "episode_index": i,
                    "success": int(i < n_succ),
                })
        frames.append(pd.DataFrame(rows, columns=EPISODES_SCHEMA))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EPISODES_SCHEMA)
    out.attrs["paired"] = False
    return out


# ----- Latency JSONs (Table R5/R7) ------------------------------------------

def load_latency_jsons(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load benchmark JSONs produced by ``bench_inference.py``.

    Expected per-file schema:
    ```json
    {
      "variant": "imf",
      "n_sampling": 1,
      "vlm_ms": [...],
      "dit_ms": [...],
      "head_ms": [...],
      "total_ms": [...],
      "hardware": "A100-64GB-bf16-FA2"
    }
    ```
    Returns a long-form DataFrame with columns
    ``[variant, n_sampling, hardware, component, latency_ms]``.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as f:
            obj = json.load(f)
        for component in ("vlm_ms", "dit_ms", "head_ms", "total_ms"):
            for ms in obj.get(component, []):
                rows.append({
                    "variant": obj["variant"],
                    "n_sampling": int(obj.get("n_sampling", 0)),
                    "hardware": obj.get("hardware", "unknown"),
                    "component": component.removesuffix("_ms"),
                    "latency_ms": float(ms),
                })
    return pd.DataFrame(rows)


def latency_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to ``mean``, ``std``, ``median`` per (variant, component)."""
    out = (
        df.groupby(["variant", "n_sampling", "component", "hardware"])["latency_ms"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    return out
