# Download Checkpoints & Hydra Configs

# These are PL ModelCheckpoint .ckpt files. They contain the raw model state_dict only
# (because training jobs were submitted with +callbacks.checkpoint.save_weights_only=True,
# so neither EMA weights nor optimizer state are persisted).
#
# Two cases:
#
# 1) Earlier iMF runs (and the imf_ablation block below) used EMA evaluation
#    (callbacks.ema.evaluate_ema_weights_instead=True). The "best" snapshot is the
#    epoch at which the EMA-evaluated avg_seq_len peaked — which, when it peaks
#    early in training, leaves the raw state_dict still close to the OpenX pretrained
#    init and visibly under-fits multi-stage suites at standalone-eval time. For
#    affected runs we prefer the run's "last.ckpt" symlink (created by save_last:
#    link in conf/callbacks/libero.yaml), which points to the most-recent epoch and
#    therefore contains the fully-trained raw weights. The "best-EMA" filenames are
#    kept in comments for traceability.
#
# 2) The Flower retrains below (Jobs 41312369/41312370/41312371/41312374 and the
#    NFE ablations 41312388/41312389/41312390) were submitted with
#    callbacks.ema.evaluate_ema_weights_instead=False, aligning with the FLOWER
#    paper's no-EMA fine-tuning protocol. The checkpoint selected by the "best"
#    metric is now the raw-weight epoch with the highest avg_seq_len, which is
#    deployment-equivalent — we point directly at it instead of last.ckpt.

# imf libero_goal (Job 40461391) — best (epoch 119, avg_seq_len ≈ 0.95)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-01/imf_40461391/saved_models/epoch=119_eval_lh/avg_seq_len=0.95.ckpt" checkpoints/imf/libero_goal/

# imf libero_goal (Job 40461391) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-01/imf_40461391/.hydra" checkpoints/imf/libero_goal/

# imf libero_spatial (Job 40461397) — best (epoch 99, avg_seq_len ≈ 0.98)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-01/imf_40461397/saved_models/epoch=99_eval_lh/avg_seq_len=0.98.ckpt" checkpoints/imf/libero_spatial/

# imf libero_spatial (Job 40461397) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-01/imf_40461397/.hydra" checkpoints/imf/libero_spatial/

# imf libero_object (Job 40461499) — last.ckpt (best-EMA was epoch 19, avg_seq_len ≈ 0.99)
rsync -avLP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-01/imf_40461499/saved_models/last.ckpt" checkpoints/imf/libero_object/

# imf libero_object (Job 40461499) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-01/imf_40461499/.hydra" checkpoints/imf/libero_object/

# imf libero_10 (Job 40461556) — best (epoch 119, avg_seq_len ≈ 0.92)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-01/imf_40461556/saved_models/epoch=119_eval_lh/avg_seq_len=0.92.ckpt" checkpoints/imf/libero_10/

# imf libero_10 (Job 40461556) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-01/imf_40461556/.hydra" checkpoints/imf/libero_10/


# imf libero_10 from-scratch ablation, default (Job 40945548) — best (epoch 49, avg_seq_len ≈ 0.90)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-06/imf_scratch_default_40945548/saved_models/epoch=49_eval_lh/avg_seq_len=0.90.ckpt" checkpoints/imf_ablation/libero_10_default/

# imf libero_10 from-scratch ablation, default (Job 40945548) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-06/imf_scratch_default_40945548/.hydra" checkpoints/imf_ablation/libero_10_default/

# imf libero_10 from-scratch ablation, heads (Job 40945552) — best (epoch 39, avg_seq_len ≈ 0.89)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-06/imf_scratch_heads_40945552/saved_models/epoch=39_eval_lh/avg_seq_len=0.89.ckpt" checkpoints/imf_ablation/libero_10_heads/

# imf libero_10 from-scratch ablation, heads (Job 40945552) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-06/imf_scratch_heads_40945552/.hydra" checkpoints/imf_ablation/libero_10_heads/

# imf libero_10 from-scratch ablation, ratio (Job 40945549) — best (epoch 109, avg_seq_len ≈ 0.93)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-06/imf_scratch_ratio_40945549/saved_models/epoch=109_eval_lh/avg_seq_len=0.93.ckpt" checkpoints/imf_ablation/libero_10_ratio/

# imf libero_10 from-scratch ablation, ratio (Job 40945549) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-06/imf_scratch_ratio_40945549/.hydra" checkpoints/imf_ablation/libero_10_ratio/

# imf libero_10 from-scratch ablation, both (Job 40945553) — best (epoch 109, avg_seq_len ≈ 0.92)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-06/imf_scratch_both_40945553/saved_models/epoch=109_eval_lh/avg_seq_len=0.92.ckpt" checkpoints/imf_ablation/libero_10_both/

# imf libero_10 from-scratch ablation, both (Job 40945553) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-06/imf_scratch_both_40945553/.hydra" checkpoints/imf_ablation/libero_10_both/

# Flower (no-EMA retrains). EMA evaluation was disabled
# (callbacks.ema.evaluate_ema_weights_instead=False) so the best-epoch raw-weight
# checkpoint is deployment-equivalent and we no longer need to fall back to last.ckpt.

# Flower libero_goal (Job 41312371) — best (epoch 109, avg_seq_len ≈ 0.97)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312371/saved_models/epoch=109_eval_lh/avg_seq_len=0.97.ckpt" checkpoints/flower/libero_goal/

# Flower libero_goal (Job 41312371) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312371/.hydra" checkpoints/flower/libero_goal/

# Flower libero_spatial (Job 41312369) — best (epoch 109, avg_seq_len ≈ 0.98)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312369/saved_models/epoch=109_eval_lh/avg_seq_len=0.98.ckpt" checkpoints/flower/libero_spatial/

# Flower libero_spatial (Job 41312369) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312369/.hydra" checkpoints/flower/libero_spatial/

# Flower libero_object (Job 41312370) — best (epoch 99, avg_seq_len ≈ 0.99)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312370/saved_models/epoch=99_eval_lh/avg_seq_len=0.99.ckpt" checkpoints/flower/libero_object/

# Flower libero_object (Job 41312370) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312370/.hydra" checkpoints/flower/libero_object/

# Flower libero_10 (Job 41312374) — best (epoch 109, avg_seq_len ≈ 0.90)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312374/saved_models/epoch=109_eval_lh/avg_seq_len=0.90.ckpt" checkpoints/flower/libero_10/

# Flower libero_10 (Job 41312374) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312374/.hydra" checkpoints/flower/libero_10/

# Flower libero_10 ablation, 1 NFE (Job 41312388) — best (epoch 109, avg_seq_len ≈ 0.88)
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312388/saved_models/epoch=109_eval_lh/avg_seq_len=0.88.ckpt" checkpoints/flower_ablation/libero_10_nfe1/

# Flower libero_10 ablation, 1 NFE (Job 41312388) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312388/.hydra" checkpoints/flower_ablation/libero_10_nfe1/

# Flower libero_10 ablation, 2 NFE (Job 41312389) — best (epoch 119, avg_seq_len ≈ 0.90).
# Best landed on the final epoch; if epoch=119_eval_lh/avg_seq_len=0.90.ckpt is missing
# on disk, fall back to last.ckpt (the final raw weights are equivalent).
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312389/saved_models/epoch=119_eval_lh/avg_seq_len=0.90.ckpt" checkpoints/flower_ablation/libero_10_nfe2/

# Flower libero_10 ablation, 2 NFE (Job 41312389) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312389/.hydra" checkpoints/flower_ablation/libero_10_nfe2/

# Flower libero_10 ablation, 3 NFE (Job 41312390) — best (epoch 119, avg_seq_len ≈ 0.87).
# Best landed on the final epoch; if epoch=119_eval_lh/avg_seq_len=0.87.ckpt is missing
# on disk, fall back to last.ckpt (the final raw weights are equivalent).
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312390/saved_models/epoch=119_eval_lh/avg_seq_len=0.87.ckpt" checkpoints/flower_ablation/libero_10_nfe3/

# Flower libero_10 ablation, 3 NFE (Job 41312390) — hydra config
rsync -avP "leonardo:/leonardo_scratch/fast/AIFAC_F02_024/project/flower_vla_calvin/logs/runs/2026-05-11/flower_41312390/.hydra" checkpoints/flower_ablation/libero_10_nfe3/
