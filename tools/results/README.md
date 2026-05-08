# `tools/results/` — thesis-results analysis pipeline

Implements the build spec in [`docs/thesis_results.md`](../../docs/thesis_results.md):
turns W&B exports + benchmark JSONs into the tables and figures cited in
the Results chapter.

```
tools/results/
├── style.py                   # Okabe–Ito palette, rcParams, formatters
├── stats.py                   # Clopper–Pearson, paired McNemar, Bonferroni, verdicts
├── variants.py                # data-driven variant ID from W&B configs
├── wandb_io.py                # CSV / JSONL / latency-JSON loaders
├── analyze_libero_results.py  # Tables R2/R3/R4/R6/R7, Figs R4/R5
├── plot_training_curves.py    # Figs R1/R2/R3, Table R1
└── README.md                  # this file
```

The benchmark binary lives separately (it needs a checkpoint + GPU, not a
W&B export):

```
flower/evaluation/bench_inference.py  # Table R5, Fig R6 source data
```

## Variants are discovered, not hardcoded

The spec names variants `rf`, `mf_naive`, `imf`, `imf_rho05`,
`imf_lhead12`, `imf_both`.  In practice the user runs ablations across
default + several `(ratio, imf_head_depth)` pairs for each model family.
`variants.identify_variant()` consumes a W&B run config and emits a
stable label:

| Family | Trigger                                              | Label                          |
| ------ | ---------------------------------------------------- | ------------------------------ |
| `rf`   | `model._target_` is `FlowerVLA`                       | `rf` (or `rf_n<N>` if `N≠4`)   |
| MF-naïve | `MeanFlowerVLA` + `use_imf=False`                  | `mf_naive`                     |
| iMF default | `MeanFlowerVLA` + `use_imf=True`, `(0.25, 8)`   | `imf`                          |
| iMF ablation | other `(ratio, imf_head_depth)`                | `imf_r{ratio:.2f}_lh{depth}`   |

Pass `--variant-labels labels.json` to remap any label to a paper-ready
display string for figure legends and table headers, e.g.
`{"imf_r0.50_lh10": "iMF, ρ=0.5, L_head=10"}`.

## Inputs

### 1 · Pretraining run-history CSVs (Figs R1/R2/R3, Table R1)

In the W&B project containing the pretraining runs:

1. Open each run.  Click **Export → Download CSV** (run-history).
2. Drop the CSVs in a directory.
3. Either invoke with inline pairs (`--csv path:variant_label`) or write a
   manifest (`--manifest manifest.json`).

Manifest schema:
```json
[
  {"csv": "exports/rf.csv",       "variant": "rf"},
  {"csv": "exports/mf_naive.csv", "variant": "mf_naive"},
  {"csv": "exports/imf.csv",      "config": {
      "model": {"_target_": "flower.models.meanflower.MeanFlowerVLA",
                "use_imf": true, "ratio": 0.25, "imf_head_depth": 8}}}
]
```

Either `variant` (label) or `config` (W&B run config dict) per entry.  If
`config` is given, `variants.identify_variant()` derives the label.

Required CSV columns (anything not matched is skipped silently):
- `_step` or `step` — global training step.
- `raw_mse` (training MSE), `val_loss/overall` (or plain `val_loss`).
- `dudt_norm`, `cos_u_v` (both families); `cos_u_utgt` (MF only); `cos_V_v` (iMF only).
- `val_loss/<dataset>` for Fig R3 — the canonical names (verbatim from the
  pretraining trainer) are listed in `style.OXE_VAL_LOSS_KEYS`.

### 2 · LIBERO per-episode outcomes (Tables R2/R3/R4/R6, Figs R4/R5)

The eval script `flower/evaluation/flower_eval_libero.py` has been patched
to write `<log_dir>/episodes.jsonl` after every run.  Each line:
```json
{"task_name": "...", "episode_index": 7, "success": 1, "suite": "libero_spatial"}
```

Tag each file with its `variant` and `epoch` either by including those
fields in the JSONL records (preferred) or by passing them as defaults to
`load_episodes_jsonl(...)` if you call the loader directly.  If you have
many runs to combine, the simplest approach is a small wrapper that walks
the eval `log_dir` tree and concatenates them with `variant`/`epoch`
filled in from the directory structure.

### 3 · Aggregate per-task SR CSV (fallback)

For pre-patch runs, the analyzer accepts a CSV with columns:
`variant, suite, epoch, task_name, success_rate`.  Episodes are expanded
to pseudo-rows (n=20 per task), but **pairing across variants is lost** —
McNemar (Table R4) and the paired forest plot (Fig R5) refuse to run on
this input.

### 4 · Inference benchmark JSONs (Table R5, Fig R6, Table R7)

Produced by `flower/evaluation/bench_inference.py`.  See the script's
docstring for invocation; outputs land in
`results/inference_bench/<run_id>.json`.

## Running

### Training-curve figures (R1/R2/R3 + Table R1)

```bash
python -m tools.results.plot_training_curves \
    --manifest exports/manifest.json \
    --out-dir thesis/figures/results/
```

Produces:
- `R1_loss_curves.pdf` — `raw_mse` (left) + `val_loss` (right), one curve per variant, with annotations for `mf_naive` divergence signatures.
- `R2_head_diagnostics.pdf` — three panels: `dudt_norm`, the head cosine (variant-appropriate), `cos_u_v`.
- `R3_val_loss_per_dataset.pdf` — facet per Open-X subset.
- `R1_stability_summary.{csv,tex}` — Table R1.

### LIBERO results (R2–R7 + Figs R4/R5)

```bash
# With the patched eval (preferred):
python -m tools.results.analyze_libero_results \
    --episodes-jsonl 'eval_logs/**/episodes.jsonl' \
    --latency-json 'results/inference_bench/*.json' \
    --reference-variant rf \
    --delta 2 5 \
    --out-dir thesis/figures/results/

# Aggregate-CSV fallback (no McNemar, no paired CIs):
python -m tools.results.analyze_libero_results \
    --episodes-csv 'eval_logs/agg_sr.csv' \
    --reference-variant rf \
    --out-dir thesis/figures/results/
```

Produces `R2_*.csv/tex`, `R3_*.csv/tex`, `R4_mcnemar_<suite>.csv/tex`,
`R4_sr_vs_epoch.pdf`, `R5_forest_noninf.pdf`, `R6_H1_verdict.csv/tex`, and
(if latency JSONs are passed) `R5_latency.csv/tex`, `R7_H2_verdict.csv/tex`.

### Inference benchmark (Table R5, Fig R6 source data)

```bash
python flower/evaluation/bench_inference.py \
    train_folder=checkpoints/imf/.hydra/config.yaml \
    checkpoint=checkpoints/imf/model.ckpt \
    bench.n_passes=200 \
    bench.n_warmup=20 \
    bench.out_json=results/inference_bench/imf.json
```

Re-run for each variant (RF, MF-naïve, iMF + each ablation), then point
`analyze_libero_results.py --latency-json` at the output directory.

## Reporting conventions (`thesis_results.md` §2)

These are enforced centrally in `style.py`:
- Decimals: SR/CIs to 0.1 pp, latency to 0.1 ms, cosines to 0.001, norms to 0.1, p-values to 3 sig figs.
- Suite-averaged CI = mean of per-task CIs (per spec §2 'CI format'), not a CI on the suite mean.
- Significance markers (`*`/`**`/`***`) reflect Bonferroni-corrected p-values; raw p in the CSV/LaTeX.
- Vector PDF output at ≥8 pt fonts.

## Verification

- **Stats unit checks**: `clopper_pearson(15, 20)` returns `[0.509, 0.913]`
  to 3 dp; `paired_mcnemar(...)` matches `statsmodels`' canonical example.
- **Variant ID smoke test**: `identify_variant({"model": {"_target_":
  "flower.models.meanflower.MeanFlowerVLA", "use_imf": True, "ratio": 0.5,
  "imf_head_depth": 10}}).label() == "imf_r0.50_lh10"`.
- **End-to-end**: hand-craft a small JSONL (4 variants × 4 suites × 20
  episodes), run the analyzer, and confirm CIs bracket point estimates and
  the forest plot whiskers cross zero where you expect.
- **Bench sanity**: the head term for iMF@1 should be smaller than RF@4,
  and the VLM term should be near-identical between variants.
