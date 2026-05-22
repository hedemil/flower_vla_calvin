# CALVIN evaluation workflow

Two distinct measurements come off a finished fine-tune run, and they live in
different places because one is hardware-independent and one is not:

| Measurement | Hardware-dependent? | Where it runs |
| --- | --- | --- |
| Success rate (avg seq len, 1000 sequences) | No | Leonardo (checkpoint already there) |
| Latency / inference time (NFE sweep) | Yes | RTX 4090 (comparable to the paper's 4090 numbers) |

## Success rate: run on Leonardo

The 300-sequence rollout every 10 epochs during training is a convergence
tracker. The headline number reported by CALVIN ABC->D / D->D (and the FLOWER
tables) is on 1000 sequences. The best EMA checkpoint is already saved on
Leonardo by `EMASafetensorsCheckpoint` (top-1 by `eval_lh/avg_seq_len`, as a
self-contained `model.safetensors` + `config.yaml` dir), so just re-evaluate it
in place:

```bash
sbatch scripts/leonardo/sbatch_eval_calvin.sh \
    $LEONARDO_FAST/project/flower_vla_calvin/logs/runs/<date>/<run_name> \
    num_sampling_steps=<NFE>
```

`sbatch_eval_calvin.sh` auto-detects the checkpoint and the dataset
(`benchmark_name` / `root_data_dir`) from the run's `.hydra/config.yaml`.

### Always pass num_sampling_steps

`sbatch_eval_calvin.sh` does not set `num_sampling_steps`, so it falls through to
`eval_calvin.yaml`'s default of 4. iMF is a 1-NFE MeanFlow model, so it must be
evaluated at 1 step; FLOWER/RF uses 4. Passing the wrong value silently
mis-benchmarks the model.

- iMF run: `num_sampling_steps=1`
- FLOWER run: `num_sampling_steps=4`

## Latency: transport weights to the 4090

Success rate does not need the 4090, but latency does, and the local machine has
little free disk, so weights go Leonardo -> HF -> 4090 without touching local.

1. Upload from the Leonardo login node (compute nodes have no internet):

   ```bash
   source $LEONARDO_WORK/venvs/flower_vla_calvin/bin/activate
   export HF_TOKEN=hf_xxx            # or: huggingface-cli login
   python scripts/leonardo/upload_checkpoint_hf.py \
       logs/runs/<date>/<run_name> --repo-id <user>/flower-vla-calvin-ckpts
   ```

   This uploads the best EMA checkpoint into a repo subfolder named after the run.

2. On the 4090, download that subfolder and run the existing latency benchmark
   (`scripts/bench_nfe_sweep.sh` / the paper-setup wrapper). Success rate is not
   re-measured here.

## This run

- Run: `imf_calvin_abcd_41936895` (iMF, ABC->D, seed 242, 160 epochs)
  - Path: `$LEONARDO_FAST/project/flower_vla_calvin/logs/runs/2026-05-19/imf_calvin_abcd_41936895`
- Best EMA checkpoint: epoch 51, `avg_seq_len = 4.42` on the 300-sequence
  in-training eval. The 1000-sequence eval gives the comparable headline figure.
- Headline SR eval command:

  ```bash
  sbatch scripts/leonardo/sbatch_eval_calvin.sh \
      $LEONARDO_FAST/project/flower_vla_calvin/logs/runs/2026-05-19/imf_calvin_abcd_41936895 \
      num_sampling_steps=1
  ```
