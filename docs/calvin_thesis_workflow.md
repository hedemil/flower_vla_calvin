# Thesis salvage plan: CALVIN D bet + LIBERO statistical reinforcement

## Context

The thesis is ~80% complete. Empirical results on LIBERO have undermined the original "iMF enables real-time deployment via NFE=1" framing for two reasons:

1. **RF works fine at NFE=1** on the LIBERO-10 fine-tuned checkpoint (cross-NFE sweep: 90.5 / 92.5 / 92.5 / 91.0% across N=1..4). The expected NFE-reduction speedup is moot — RF doesn't need extra steps.
2. **Latency parity at NFE=1.** From `docs/figures/results/R5_latency.csv`: RF@NFE=1 = 111.99 ms, iMF@NFE=1 = 111.72 ms. The published 1.44× speedup is purely RF@NFE=4 → iMF@NFE=1 — a sampling-schedule effect, not architectural.

The likely cause is **dataset contamination**: `flowereef` pretraining includes `libero_10_no_noops` (weight 10.0) and `libero_goal_no_noops` (weight 18.0), so fine-tuning on LIBERO is in-distribution adaptation. The velocity field is already well-shaped on these tasks, so coarse Euler integration recovers it. The thesis itself acknowledges this in `docs/thesis/thesis_method.md`.

iMF still wins +5pp on LIBERO-Long-10 at matched NFE=1, but McNemar shows that gap is statistically borderline (see §"What's done" below).

**Intended outcome with <2 weeks:** run one paired CALVIN D fine-tune (RF and iMF from matched pretrained checkpoints) as the primary bet. CALVIN is genuinely OOD to `flowereef`. Within CALVIN, choose D over ABC→D because:

- FLOWER paper Table 9 shows the *largest* baseline-vs-2nd-best margin on CALVIN D (+10pp absolute, +13% relative) vs ABC (+4.8pp) and ABCD (+3.0pp) — D is the most unsaturated of the three, the most likely place for an objective change to register.
- task_D_D dataset is ~165 GB compressed (~250–300 GB extracted) vs ~700 GB for task_ABC_D, leaving compute/storage budget for actual fine-tuning.
- CALVIN D is still fully OOD to the pretraining mix.

If CALVIN D delivers, add ABC→D as bonus generalization evidence. If CALVIN underdelivers, fall back to the honest negative-result reframing.

---

## What's done already

### Track A1 — Paired McNemar on LIBERO (`docs/figures/results/R4_mcnemar_libero.csv`)

**iMF vs RF at default NFE (RF=4, iMF=1):**

| Suite | n | RF k | iMF k | Δpp | n11/n10/n01/n00 | p (exact) | Bonferroni 0.0125 |
|---|---|---|---|---|---|---|---|
| libero_spatial | 200 | 198 | 196 | −1.0 | 194/2/4/0 | 0.6875 | NS |
| libero_object | 200 | 196 | 192 | −2.0 | 189/3/7/1 | 0.3438 | NS |
| libero_goal | 200 | 190 | 192 | +1.0 | 182/10/8/0 | 0.8145 | NS |
| **libero_10** | **200** | **182** | **192** | **+5.0** | **176/16/6/2** | **0.0525** | **NS (just misses uncorrected 0.05)** |

**iMF vs RF at matched NFE=1 (libero_10 only):**

| Comparison | Δpp | p |
|---|---|---|
| imf vs rf_n1 | +5.5 | 0.027 (uncorrected sig; Bonferroni NS) |
| imf vs rf_n2 | +3.5 | 0.143 |
| imf vs rf_n3 | +3.5 | 0.143 |

**Implication.** The +5pp Long-10 win is directional but underpowered (only 22 discordant pairs). Cannot defend "iMF significantly improves Long-10" at Bonferroni — only "iMF directionally improves Long-10." CALVIN must carry the headline.

### Track B1 — eval-during-training instrumentation (done locally)

- `flower/rollout/rollout_long_horizon.py`: now writes `rollout_episodes.jsonl` per seed run dir. One chain record + 5 subtask records per sequence (`kind`, `epoch`, `sequence_index`, `eval_sequence`, `success_counter`, per-subtask name/success/steps).
- `flower/evaluation/flower_evaluate.py`: same JSONL emission for standalone post-training eval.
- Per-subtask SR also logged to wandb at every eval epoch.
- Sequence indices made globally unique across DDP ranks.

### Scripts in place

| Path | Purpose |
|---|---|
| `scripts/leonardo/sbatch_finetune_calvin.sh` | Paired RF/iMF CALVIN D fine-tune. Defaults: seed 242, 160 epochs (=40k opt steps), num_sequences=300 during training, eval every 10 epochs. |
| `scripts/leonardo/sbatch_eval_calvin.sh` | Post-training final eval at num_sequences=1000 on the best checkpoint. Auto-detects train_folder + ckpt. |
| `scripts/leonardo/setup_leonardo.sh` | CALVIN download. Now parametrised: defaults to task_D_D, override with `CALVIN_TASK=task_ABC_D` for cross-env. |
| `tools/results/mcnemar_libero.py` | Paired McNemar across LIBERO suites (already run, see Track A1). |
| `tools/results/mcnemar_calvin.py` | Paired McNemar + chain-length analysis on CALVIN `rollout_episodes.jsonl`. |

### Decisions deferred

- **A2 (steps-to-success on co-successful episodes)** — useful but won't change the framing decision. Deferred.
- **A3 (full NFE latency grid)** — data already exists in `docs/figures/results/R5_latency.csv`. Surface it in the thesis text; no re-run needed.
- **A4 (broken iMF ablations)** — `libero_*_imf_heads/ratio/both_evaluation` show 0–3% SR but training-time evals on same checkpoints showed 88–90%. Almost certainly a checkpoint-loading bug in the post-hoc eval pipeline. Decision: **report training-time numbers with explicit methodological note** in the thesis; do not publish the 0–3% numbers as real.
- **A5 (libero_spatial NFE sweep)** — would test contamination explanation cleanly. Deferred unless time permits after CALVIN.

---

## How CALVIN evaluation works (read once)

The 1000 "chains" are 1000 randomly-sampled sequences of 5 tasks each. The agent must do them in order; it doesn't see task N+1 until it solves task N. Failure at task K stops the chain — all later tasks score 0.

| Paper metric | Meaning | wandb key |
|---|---|---|
| "1 instruction in a row" | Fraction of chains where ≥1 task solved | `eval_lh/sr_chain_1` |
| "5 in a row" | Fraction of chains where all 5 solved | `eval_lh/sr_chain_5` |
| "Avg. Len." | Mean chain length completed (0–5) | `eval_lh/avg_seq_len` |

Table 9's "87.0%" for CALVIN D = `avg_seq_len / 5`. FLOWER's D→D target: avg_len ≈ **4.35**, sr_chain_5 ≈ **0.749**, sr_chain_1 ≈ **0.974**.

**Where iMF could win that RF can't:** if both reach similar `avg_seq_len` but iMF has higher `sr_chain_5`, the objective decomposes long horizons better — exactly the story we want.

---

## End-to-end workflow

### Step 1 — Cluster setup (you on Leonardo, ~2 hours)

```bash
# On login node
ssh ehed0000@login.leonardo.cineca.it

# Check quota
lfs quota -h -u $USER /leonardo_work

# Sync local changes to FAST scratch (from your laptop)
rsync -av \
  flower/rollout/rollout_long_horizon.py \
  flower/evaluation/flower_evaluate.py \
  tools/results/mcnemar_calvin.py \
  scripts/leonardo/sbatch_finetune_calvin.sh \
  scripts/leonardo/sbatch_eval_calvin.sh \
  scripts/leonardo/setup_leonardo.sh \
  leonardo:$LEONARDO_FAST/project/flower_vla_calvin/...

# Download CALVIN D (~165 GB compressed, ~250-300 GB extracted)
# setup_leonardo.sh now uses `wget -c` (resumable) and verifies zip integrity.
cd $LEONARDO_FAST/project/flower_vla_calvin
./scripts/leonardo/setup_leonardo.sh calvin

# Verify pretrained checkpoints exist
ls $LEONARDO_WORK/checkpoints/pretrained/flower_baseline_290000.safetensors
ls $LEONARDO_WORK/checkpoints/pretrained/imf_checkpoint_290000.safetensors
```

### Step 2 — Launch paired fine-tunes (you on Leonardo, ~1 minute)

```bash
sbatch scripts/leonardo/sbatch_finetune_calvin.sh flower
sbatch scripts/leonardo/sbatch_finetune_calvin.sh imf
```

Both runs use the same seed (242), each loads its own matched pretrained backbone (matches `sbatch_train_libero*.sh` convention). 160 epochs ≈ 40k optimizer steps, eval at epochs 1, 11, 21, …, 161 with `num_sequences=300`. Wall time per run: ~12–15 h on 4×A100. Both run in parallel → finishes ~1 day later.

### Step 3 — Monitor during training (you, periodically)

```bash
# As soon as the first epoch-1 eval drops on each run:
python -m tools.results.mcnemar_calvin \
  --rf-jsonl  $LEONARDO_FAST/project/flower_vla_calvin/logs/runs/<DATE>/flower_calvin_d_<JOB>/seed_242/rollout_episodes.jsonl \
  --imf-jsonl $LEONARDO_FAST/project/flower_vla_calvin/logs/runs/<DATE>/imf_calvin_d_<JOB>/seed_242/rollout_episodes.jsonl
```

What to watch:

- `sr_chain_1` curve — should hit ≥0.90 within ~5–10k steps for both models.
- `sr_chain_5` and `avg_seq_len` — these are the discriminators. iMF advantage (if real) shows up at chain ≥4 and ≥5.
- Δpp at chain ≥5 with positive sign across multiple epochs → real signal.
- If both plateau by epoch ~60 with `sr_chain_5` < 0.5, kill and reconsider; more training won't help.

### Step 4 — Skip the final 1000-sequence eval (deliberate)

**Originally planned but deprecated:** the `rollout_episodes.jsonl` written during
training already captures **EMA-eval performance** for every intermediate eval
epoch (eval-during-training uses EMA weights via `evaluate_ema_weights_instead=True`).
300 paired chains × 16 eval epochs × 2 models = ~10k paired observations — far
more statistical power than a single 1000-chain post-hoc pass, and not subject
to whatever PL's `save_weights_only=True` does to the EMA callback state.

The `sbatch_eval_calvin.sh` script exists if you want it for a bonus
ABC→D run, but is not needed for the headline thesis result.

### Step 5 — Headline McNemar + thesis numbers (from in-training JSONL)

Pick the best epoch per run (`argmax(avg_seq_len)`) from the in-training JSONL,
then run the paired analysis. The `--epoch` flag defaults to the latest epoch in
each file but you can pin a specific one for a fair comparison:

```bash
# Default: latest epoch in each file
python -m tools.results.mcnemar_calvin \
  --rf-jsonl  $LEONARDO_FAST/.../flower_calvin_d_<JOB>/seed_242/rollout_episodes.jsonl \
  --imf-jsonl $LEONARDO_FAST/.../imf_calvin_d_<JOB>/seed_242/rollout_episodes.jsonl \
  --out       docs/figures/results/R_calvin_d_mcnemar.csv

# Or pin a specific eval epoch (e.g., both models at epoch 161 - the final eval)
python -m tools.results.mcnemar_calvin --rf-jsonl ... --imf-jsonl ... --epoch 161
```

Output includes per-chain-length Δpp + Newcombe CI + Bonferroni-corrected p-values, plus the Table 9-comparable `avg_seq_len / 5` headline number for both models.

**Headline framing for the thesis:** "iMF avg_seq_len = X.XX on CALVIN D over 300 paired chains at epoch Y; RF = Y.YY (paired McNemar p = ...). The 300-chain sample is below the FLOWER paper's 1000-chain protocol but the paired test does not require matching the unpaired baseline's sample size."

---

## Decision matrix for thesis framing

After CALVIN results land, pick the row that matches the evidence:

| Track A McNemar (LIBERO) | Track B CALVIN result | Thesis framing |
|---|---|---|
| Significant | iMF wins on CALVIN D | **OOD generalization story.** Headline: iMF objective improves OOD adaptation. LIBERO confirms with a smaller (in-distribution) gap. |
| Significant | iMF ties / small gap | **Matched-compute objective advantage story.** LIBERO is the headline; CALVIN supports weakly. |
| **Borderline (current)** | iMF wins on CALVIN D | **CALVIN OOD story.** LIBERO saturation explained by pretraining contamination. Thesis pivots benchmark. |
| **Borderline (current)** | iMF ties / loses | **Honest negative-result thesis.** Document everything: NFE saturation, latency parity, directional LIBERO win, CALVIN tie. Reframe contribution: "we built and rigorously evaluated iMF for VLAs; the expected advantages do not materialise in this regime, and we identify why." Defensible, lower-ceiling, passes. |

The framing in Ch 6/7 is written **after** the numbers are in. Do not pre-commit to a story the evidence may not support.

---

## Examiner attack surface — anticipated questions and answers

| Attack | Answer |
|---|---|
| "Your latency speedup vanishes at matched NFE." | Yes — we report this directly in Ch 5 Table X. The thesis contribution is the per-step quality advantage at matched compute, demonstrated on OOD data (CALVIN). |
| "LIBERO is in your pretraining mix — fine-tuning is trivial." | Acknowledged in §3.2. That's exactly why we evaluate on CALVIN D, which is OOD. The CALVIN gap is the real test. |
| "Why didn't you train RF natively at NFE=1?" | Out of compute budget. Future work. The iMF advantage at matched NFE=1 still stands. (Strongest remaining attack — be ready.) |
| "Is Long-10 +5pp significant?" | McNemar exact two-sided p=0.0525, uncorrected. Misses both uncorrected α=0.05 and Bonferroni-corrected α=0.0125. We report it as a directional result, not a definitive one. |
| "Your hyperparameter ablations show 0%." | Checkpoint-loading bug in post-hoc eval pipeline; training-time evals on same checkpoints showed 88–90%. We report training-time numbers and document the discrepancy. |
| "Only one CALVIN seed — could be noise." | Honest answer: time constraint. Report it explicitly. Lower confidence is the cost of running one seed. |
| "Why CALVIN D and not the harder ABC→D?" | CALVIN D has the largest unsaturated headroom in FLOWER Table 9 (+10pp baseline-vs-2nd-best vs +4.8pp on ABC); is fully OOD to flowereef pretraining; fits compute budget. ABC→D listed as future work. |

---

## Critical files

**Code modified this branch:**

- `flower/rollout/rollout_long_horizon.py` — per-sequence + per-subtask JSONL emission
- `flower/evaluation/flower_evaluate.py` — JSONL emission for standalone eval
- `scripts/leonardo/setup_leonardo.sh` — parametrised CALVIN task
- `scripts/leonardo/sbatch_finetune_calvin.sh` — new, paired RF/iMF
- `scripts/leonardo/sbatch_eval_calvin.sh` — new, post-training 1000-seq eval
- `tools/results/mcnemar_libero.py` — new, LIBERO paired McNemar (already run)
- `tools/results/mcnemar_calvin.py` — new, CALVIN paired McNemar + chain-length

**Data / cluster:**

- Leonardo paths: `$WORK = /leonardo_work/AIFAC_F02_024`, `$FAST = /leonardo_scratch/fast/AIFAC_F02_024`
- Existing LIBERO data: `$WORK/data/libero/`
- New CALVIN data: `$WORK/data/calvin/task_D_D` (after Step 1)
- Pretrained checkpoints: `$WORK/checkpoints/pretrained/{flower_baseline,imf_checkpoint}_290000.safetensors`
- Existing latency CSV: `docs/figures/results/R5_latency.csv` (already has RF@NFE=1)
- McNemar CSV: `docs/figures/results/R4_mcnemar_libero.csv`

**Thesis files (edit only after CALVIN results land):**

- `docs/thesis/thesis_intro.md` — H₁/H₂ phrasing. Either add a revised-hypothesis subsection or rewrite to match evidence. Do **not** silently retrofit.
- `docs/thesis/thesis_results_draft.md` — add §C-CALVIN, latency grid, McNemar tables
- `docs/thesis/thesis_method.md` — extend §3.2 to describe CALVIN fine-tuning protocol
- Ch 6 (Discussion), Ch 7 (Conclusions) — write last, after the matrix decision

---

## What I (Claude) will not do

- Cherry-pick seeds, suites, chain lengths, or metrics.
- Run statistical tests until one's significant.
- Hide the cross-NFE saturation finding or the RF@NFE=1 latency parity.
- Silently retrofit H₁/H₂ in the intro chapter to match the post-hoc story.
- Report broken ablations (the 0–3% post-hoc evals) as real iMF fragility.

All detectable, all degree-risking.

---

## Verification checklist

- [ ] CALVIN D downloaded to `$WORK/data/calvin/task_D_D`
- [ ] Both pretrained checkpoints exist under `$WORK/checkpoints/pretrained/`
- [ ] `sbatch scripts/leonardo/sbatch_finetune_calvin.sh flower` submitted
- [ ] `sbatch scripts/leonardo/sbatch_finetune_calvin.sh imf` submitted
- [ ] First epoch-1 eval JSONL exists for both runs; McNemar at chain ≥5 directionally positive
- [ ] Both fine-tunes completed (best checkpoint saved per run)
- [ ] `sbatch_eval_calvin.sh` run on each; `final_eval_*/logs/<ts>/rollout_episodes.jsonl` exists
- [ ] Final McNemar table written to `docs/figures/results/R_calvin_d_mcnemar.csv`
- [ ] Matrix decision made; thesis framing chosen based on evidence
- [ ] Ch 6 / Ch 7 drafted
- [ ] Practice defense: every row in the attack table has an answer that doesn't require hedging
- [ ] One-line thesis claim, true given evidence, survives hostile reading
