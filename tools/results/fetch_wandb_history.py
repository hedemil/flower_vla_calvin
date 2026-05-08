"""Fetch full run histories from W&B and write CSVs the analyzer can consume.

Replaces the manual "Export → CSV" UI step in ``docs/run_analysis.md`` §5a for
runs whose history is only reachable via the W&B Python API. Output schema
matches what ``tools/results/wandb_io.load_run_history`` expects, so no
analyzer changes are needed.

Usage::

    python -m tools.results.fetch_wandb_history \\
      --out-dir wandb_exports/ \\
      fz9telzf:rf u4ps8iva:imf

Each positional argument is ``<run_spec>:<variant_label>``. ``run_spec`` may
be:

* a bare run ID (``fz9telzf``) — uses ``--entity``/``--project`` defaults,
* ``<entity>/<project>/<run_id>``, or
* ``<entity>/<project>/runs/<run_id>`` (URL-style; ``runs/`` is stripped).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import style

LOG = logging.getLogger("fetch_wandb_history")

DEFAULT_ENTITY = "VLA-Thesis"
DEFAULT_PROJECT = "OXE_pretraining"

# Scalars the §3 analyzer reads, plus close cousins seen in these runs.
# Restricting `scan_history(keys=...)` to this set is ~10x faster than
# fetching every metric the trainer logs (gradients, LR, system stats, etc.).
# Keys absent from a given run are silently skipped by the W&B API.
ANALYZER_KEYS: list[str] = sorted({
    # Fig R1 left (training loss) — RF logs `loss`, iMF logs `raw_mse_vc`/`raw_mse_V`.
    "raw_mse",
    "raw_mse_vc",
    "raw_mse_V",
    "loss",
    # Fig R1 right (val loss).
    "val_loss",
    "val_loss/overall",
    # Fig R2 head diagnostics.
    "dudt_norm",
    "cos_u_utgt",
    "cos_V_v",
    "cos_u_v",
    # Fig R3 per-dataset val loss.
    *style.OXE_VAL_LOSS_KEYS.values(),
    # Trainer-only flags. Used by --align to filter out per-batch FLOWER rows
    # (those have `loss` but no learning_rate / test_loss).
    "learning_rate",
    "dit_learning_rate",
    "vlm_learning_rate",
    "test_loss",
})

# Columns that are ONLY logged from the trainer-stream tick (never from a
# per-batch hook). Filtering rows on the first column present here gives us
# the canonical "one row per LOG_INTERVAL training steps" view.
TRAINER_FLAG_CANDIDATES: tuple[str, ...] = (
    "dit_learning_rate",
    "vlm_learning_rate",
    "learning_rate",
    "test_loss",
)


def align_to_training_steps(
    df: pd.DataFrame, *, log_interval: int, variant: str,
) -> pd.DataFrame:
    """Reindex `_step` so it represents training step, comparable across runs.

    Two row classes survive:
      * Trainer-stream rows — those where a trainer-only column (e.g.
        `dit_learning_rate`) is non-null. Each such row k gets
        `training_step = k * log_interval` (anchor positions).
      * Validation rows — any row with a `val_loss*` column non-null. These
        don't carry the trainer flag (validation logs don't include LR), so
        their `training_step` is interpolated from neighbouring anchors via
        `np.interp` on `_wandb_step`.

    Per-batch FLOWER rows (only `loss` populated, no trainer flag, no val
    key) are dropped — that's the noise the recipe filter targets.

    The original W&B counter is preserved as `_wandb_step`; `_step` is
    overwritten so the analyzer's existing `_normalize_step_column` routes
    the aligned counter into every plot.
    """
    df = df.sort_values("_step").reset_index(drop=True)
    flag = next(
        (c for c in TRAINER_FLAG_CANDIDATES if c in df.columns and df[c].notna().any()),
        None,
    )

    if flag is None:
        LOG.warning(
            "  variant=%s: no trainer-flag column found among %s — assuming "
            "every row is trainer-stream and reindexing as-is.",
            variant, TRAINER_FLAG_CANDIDATES,
        )
        df["_wandb_step"] = df["_step"]
        df["_step"] = df.index * log_interval
    else:
        anchors = df[df[flag].notna()].reset_index(drop=True)
        anchor_wandb = anchors["_step"].to_numpy()
        anchor_train = np.arange(len(anchors)) * log_interval

        val_cols = [c for c in df.columns if c.startswith("val_loss")]
        is_trainer = df[flag].notna()
        is_val = (
            df[val_cols].notna().any(axis=1) if val_cols else pd.Series(False, index=df.index)
        )
        keep = is_trainer | is_val
        n_total = len(df)
        n_train = int(is_trainer.sum())
        n_val = int((is_val & ~is_trainer).sum())
        df = df[keep].reset_index(drop=True)

        df["_wandb_step"] = df["_step"]
        df["_step"] = np.interp(
            df["_wandb_step"].to_numpy(), anchor_wandb, anchor_train,
        ).astype(int)

        LOG.info(
            "  variant=%s: kept %d trainer-stream + %d val rows out of %d "
            "(flag=%r, log_interval=%d)",
            variant, n_train, n_val, n_total, flag, log_interval,
        )

    if "raw_mse" not in df.columns and "loss" in df.columns:
        df["raw_mse"] = df["loss"]
        LOG.info("  variant=%s: aliased loss -> raw_mse (Fig R1 left)", variant)

    return df


def parse_run_spec(spec: str, *, entity: str, project: str) -> tuple[str, str]:
    """Return ``(api_path, variant_label)`` from ``<run_spec>:<variant>``."""
    if ":" not in spec:
        raise SystemExit(f"expected '<run_spec>:<variant>', got {spec!r}")
    run_spec, _, variant = spec.partition(":")
    if not variant:
        raise SystemExit(f"empty variant label in {spec!r}")

    parts = [p for p in run_spec.split("/") if p and p != "runs"]
    if len(parts) == 1:
        api_path = f"{entity}/{project}/{parts[0]}"
    elif len(parts) == 3:
        api_path = "/".join(parts)
    else:
        raise SystemExit(
            f"run spec {run_spec!r} should be '<run_id>' or "
            "'<entity>/<project>/<run_id>'"
        )
    return api_path, variant


def _history_with_retry(
    run, key: str, *, samples: int, attempts: int = 4, base_wait: float = 30.0,
) -> pd.DataFrame | None:
    """`run.history(keys=[key])` with exponential backoff on transient errors.

    W&B's GraphQL backend occasionally returns 5xx mid-fetch; the SDK's
    own retry gives up after ~30s. Retrying with longer waits (30s, 60s,
    120s, 240s) gets through almost every flake.
    """
    for attempt in range(1, attempts + 1):
        try:
            return run.history(
                keys=[key], samples=samples, pandas=True, x_axis="_step",
            )
        except Exception as e:
            if attempt == attempts:
                LOG.error("  key %r: gave up after %d attempts (%s)", key, attempts, e)
                raise
            wait = base_wait * (2 ** (attempt - 1))
            LOG.warning(
                "  key %r: attempt %d/%d failed (%s) — retrying in %.0fs",
                key, attempt, attempts, e, wait,
            )
            time.sleep(wait)
    return None


def fetch_one(
    api,
    api_path: str,
    *,
    keys: list[str] | None,
    page_size: int,
    samples: int,
    log_every: int,
) -> pd.DataFrame:
    run = api.run(api_path)
    LOG.info(
        "  run name=%s, state=%s, last_step=%s",
        run.name, run.state, getattr(run, "lastHistoryStep", "?"),
    )

    # All-keys path: stream the firehose.
    if keys is None:
        rows: list[dict] = []
        for row in run.scan_history(page_size=page_size):
            rows.append(row)
            if log_every and len(rows) % log_every == 0:
                LOG.info("  ... %d rows fetched", len(rows))
        if not rows:
            raise SystemExit(f"run {api_path} has no logged history.")
        LOG.info("  done: %d rows", len(rows))
        return pd.DataFrame(rows)

    # Per-key path: use run.history(keys=[k], samples=N), which always
    # returns a DataFrame containing _step + the key (unlike
    # scan_history(keys=[...]) which drops _step in the sampled-history
    # query and ALSO requires every key to be non-null in the same row).
    # `samples` caps the row count; pick it >= the run's last step.
    merged: pd.DataFrame | None = None
    for k in keys:
        df = _history_with_retry(run, k, samples=samples)
        if df is None or df.empty:
            LOG.info("  key %r: 0 rows (skipping)", k)
            continue
        if k not in df.columns:
            LOG.info(
                "  key %r: column missing in returned df (cols=%s)",
                k, list(df.columns),
            )
            continue
        sub = df[["_step", k]].dropna(subset=[k]).drop_duplicates("_step")
        if sub.empty:
            LOG.info("  key %r: all values NaN (skipping)", k)
            continue
        merged = sub if merged is None else merged.merge(sub, on="_step", how="outer")
        LOG.info("  key %r: %d rows", k, len(sub))

    if merged is None or merged.empty:
        raise SystemExit(
            f"run {api_path} returned no rows for any of keys={keys!r} — "
            "rerun with --all-keys to see what's actually logged."
        )
    merged = merged.sort_values("_step").reset_index(drop=True)
    LOG.info("  merged: %d rows × %d columns", len(merged), merged.shape[1])
    return merged


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="`<run_spec>:<variant>` pairs.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="W&B GraphQL request timeout in seconds (default: 120). "
        "Long pretraining histories need >>19s (the wandb default).",
    )
    p.add_argument(
        "--all-keys",
        action="store_true",
        help="Fetch every logged scalar instead of just the analyzer's "
        "subset. Much slower but useful for debugging.",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=10_000,
        help="Rows per W&B GraphQL page for --all-keys mode (default: 10000).",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=500_000,
        help="Max rows per key for the targeted-keys path (default: 500000). "
        "Set this above the run's last step to avoid downsampling.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10_000,
        help="Print progress every N rows fetched (default: 10000; 0 = silent).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants whose <out-dir>/<variant>.csv already exists. "
        "Useful for resuming after a transient W&B 5xx mid-fetch.",
    )
    p.add_argument(
        "--no-align",
        action="store_true",
        help="Skip the post-fetch alignment pass. Default: align `_step` to "
        "actual training step (filters per-batch FLOWER rows, scales by "
        "--log-interval) so plots are comparable across variants.",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Trainer-stream log cadence in actual training steps "
        "(default: 100, matches flower_trainer.py:366).",
    )
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing manifest.json alongside the CSVs.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import wandb

    api = wandb.Api(timeout=args.timeout)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keys = None if args.all_keys else ANALYZER_KEYS
    LOG.info("Fetching keys=%s", "<all>" if keys is None else keys)

    manifest: list[dict[str, str]] = []
    for spec in args.runs:
        api_path, variant = parse_run_spec(spec, entity=args.entity, project=args.project)
        csv_path = args.out_dir / f"{variant}.csv"
        if args.skip_existing and csv_path.exists():
            LOG.info("Skipping %s — %s exists (--skip-existing)", variant, csv_path)
            manifest.append({"csv": csv_path.name, "variant": variant})
            continue
        LOG.info("Fetching %s as variant=%s", api_path, variant)
        df = fetch_one(
            api,
            api_path,
            keys=keys,
            page_size=args.page_size,
            samples=args.samples,
            log_every=args.log_every,
        )
        if not args.no_align:
            df = align_to_training_steps(
                df, log_interval=args.log_interval, variant=variant,
            )
        df.to_csv(csv_path, index=False)
        LOG.info("Wrote %s (%d rows, %d columns)", csv_path, len(df), df.shape[1])
        manifest.append({"csv": csv_path.name, "variant": variant})

    if not args.no_manifest:
        manifest_path = args.out_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        LOG.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
