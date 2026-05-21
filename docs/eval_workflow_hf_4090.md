# CALVIN headline eval workflow: Leonardo to HF to RTX 4090

Training runs on Leonardo (A100). The headline `num_sequences=1000` CALVIN
evaluation runs on a separate RTX 4090 box reached over ssh. The local machine
has little free disk, so the checkpoint never touches it: it goes straight from
Leonardo to the Hugging Face Hub, and from the Hub to the 4090.

```
Leonardo (train + upload)  ->  HF Hub (private repo)  ->  RTX 4090 (download + eval)
```

## Why a 1000-sequence eval at all

During training the rollout callback evaluates every 10 epochs on 300 sequences.
That is a convergence tracker, not the headline number. The CALVIN ABC->D and
D->D benchmarks (and the FLOWER paper tables) report on 1000 sequences, so the
final comparable figure comes from re-running the best checkpoint at 1000.

The best EMA checkpoint is already kept on Leonardo: `EMASafetensorsCheckpoint`
keeps the top-1 by `eval_lh/avg_seq_len` as a self-contained HF-format dir
(`model.safetensors` + `config.yaml`). Success rate is hardware independent, so
running it on the 4090 instead of Leonardo changes nothing in the metric; it is
purely a logistics choice. (Latency and NFE-sweep numbers, which do depend on
hardware, stay on their dedicated benchmark setup.)

## Step 1: upload from the Leonardo login node

Compute nodes have no internet; the login node does. Run there:

```bash
source $LEONARDO_WORK/venvs/flower_vla_calvin/bin/activate
export HF_TOKEN=hf_xxx            # or: huggingface-cli login
cd $LEONARDO_FAST/project/flower_vla_calvin

python scripts/leonardo/upload_checkpoint_hf.py \
    logs/runs/2026-05-19/imf_calvin_abcd_41936895 \
    --repo-id <user>/flower-vla-calvin-ckpts
```

The script finds the highest-scoring `saved_models/*/model.safetensors` under the
run dir and uploads it (plus `config.yaml`) into a repo subfolder named after the
run (here `imf_calvin_abcd_41936895`). The repo is private by default; pass
`--public` to override. One repo holds many runs.

## Step 2: download and evaluate on the 4090

Over ssh on the 4090 box:

```bash
cd $FLOWER_CODE_DIR          # where the repo is cloned, default ~/flower_vla_calvin
source <venv>/bin/activate

CALVIN_DATA_DIR=/data/calvin/task_ABC_D \
scripts/rtx4090/eval_calvin_from_hf.sh \
    <user>/flower-vla-calvin-ckpts imf_calvin_abcd_41936895
```

The script downloads only that subfolder, then runs
`flower/evaluation/flower_evaluate.py` at `num_sequences=1000`. Output (per
sequence and per subtask) lands in
`ckpts/<subfolder>/eval_<timestamp>/logs/<ts>/rollout_episodes.jsonl`.

### NFE is set automatically, but verify it

`eval_calvin.yaml` forces `model.num_sampling_steps` from the top-level
`num_sampling_steps` (default 4) through `eval_cfg_overwrite`. iMF is a 1-NFE
MeanFlow model, so it must run at 1 step. The script infers NFE from the name
(`imf` -> 1, otherwise 4) and echoes it in the header. Override explicitly if
the name does not carry the model:

```bash
# force 1 NFE
scripts/rtx4090/eval_calvin_from_hf.sh <repo> <subfolder> 1
```

## Environment knobs (4090 script)

| Variable          | Default                  | Meaning                                   |
| ----------------- | ------------------------ | ----------------------------------------- |
| `CALVIN_DATA_DIR` | (required)               | CALVIN task dir (`task_ABC_D`, `task_D_D`)|
| `FLOWER_CODE_DIR` | `~/flower_vla_calvin`    | repo location on the 4090                 |
| `CKPT_ROOT`       | `$FLOWER_CODE_DIR/ckpts` | where checkpoints download                |
| `DEVICE`          | `0`                      | CUDA device index                         |
| `NUM_SEQUENCES`   | `1000`                   | rollout chains                            |

## Prerequisites on the 4090 (one time)

The eval needs the repo, a venv with the deps and the CALVIN env, the task
dataset, and internet (for the checkpoint and Florence-2-large). If the dataset
is missing, fetch the relevant `task_ABC_D` / `task_D_D` split as on Leonardo.
Florence-2-large downloads automatically on first run (no offline flags here,
unlike the Leonardo compute nodes).

## This run

- Run: `imf_calvin_abcd_41936895` (iMF, ABC->D, seed 242, 160 epochs)
- Best EMA checkpoint: epoch 51, `avg_seq_len = 4.42` on the 300-sequence
  in-training eval; the 1000-sequence eval gives the comparable headline figure.
- NFE: 1.
