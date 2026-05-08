"""Wall-clock benchmark of the VLM + DiT + head pipeline
(`docs/thesis_results.md` §5).

Decomposes total inference latency into three components and writes the
per-pass measurements to a JSON file so ``analyze_libero_results.py`` can
build Table R5 / Fig. R6 / Table R7.

Usage (Hydra; mirrors flower_eval_libero.py):

    python flower/evaluation/bench_inference.py \\
        train_folder=path/to/.hydra/config.yaml \\
        checkpoint=path/to/model.ckpt \\
        bench.n_passes=200 \\
        bench.n_warmup=20 \\
        bench.out_json=results/inference_bench/imf.json

The script reuses ``get_default_mode_and_env`` from the LIBERO eval to load
exactly the same model the rollout script would run, so the timing applies
to the production code path.  All measurements use ``torch.cuda.Event``,
which is the only PyTorch API that produces accurate GPU wall-clock times
(``time.perf_counter`` measures dispatch overhead, not kernel execution).

Why decompose: the speed-up claim is "the head got cheaper", not
"everything got cheaper".  VLM dominates total latency and is variant-
invariant — without separation we cannot attribute the gain.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

# Ensure the local repo is importable when run via slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from flower.evaluation.utils import get_default_mode_and_env  # noqa: E402


LOG = logging.getLogger("bench_inference")


def _cuda_time(fn, *, n_passes: int, n_warmup: int) -> np.ndarray:
    """Time `fn` over ``n_passes`` calls (after ``n_warmup``) using cuda events."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_passes)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_passes)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return np.array([s.elapsed_time(e) for s, e in zip(starts, ends)], dtype=float)


def _make_synthetic_obs(model, device: torch.device) -> tuple[Dict, Dict]:
    """Build synthetic ``(obs, goal)`` matching the LIBERO eval call site
    (`flower_eval_libero.py:295-328`).  Shapes: rgb [1,1,3,224,224]; lang str.
    """
    dtype = next(model.parameters()).dtype
    rgb_static = torch.rand(1, 1, 3, 224, 224, device=device, dtype=dtype)
    rgb_gripper = torch.rand(1, 1, 3, 224, 224, device=device, dtype=dtype)
    obs = {
        "rgb_obs": {
            "rgb_static": rgb_static,
            "rgb_gripper": rgb_gripper,
        },
        "robot_obs": torch.zeros(1, 7, device=device, dtype=dtype),
        "gripper_states": torch.zeros(1, 2, device=device, dtype=dtype),
        "depth_obs": {},
    }
    goal = {"lang_text": "pick up the object and place it on the plate", "lang": None}
    return obs, goal


def _bench_components(model, obs: Dict, goal: Dict, *, n_passes: int, n_warmup: int) -> Dict[str, np.ndarray]:
    """Time VLM, head (full sample loop), single DiT step, and total step."""
    model.eval()
    # Build the batch the way `forward()` does (flower.py:828-839) so the VLM
    # call is identical.
    rgb_static = obs["rgb_obs"]["rgb_static"]
    rgb_gripper = obs["rgb_obs"]["rgb_gripper"]
    batch = {
        "rgb_obs": {"rgb_static": rgb_static, "rgb_gripper": rgb_gripper},
        "lang_text": [goal["lang_text"]],
    }

    @torch.no_grad()
    def fn_vlm():
        return model.encode_observations(batch)

    # Warm-cache features for downstream timing
    with torch.no_grad():
        features = model.encode_observations(batch)
    noise = torch.randn(
        len(features["features"]),
        model.act_window_size,
        model.action_dim,
        device=features["features"].device,
        dtype=features["features"].dtype,
    )
    # Single-step DiT timing: t=1.0, full noise input
    t_tensor = torch.full((noise.shape[0],), 1.0, device=noise.device)

    @torch.no_grad()
    def fn_dit_single():
        if hasattr(model, "dit_forward"):
            return model.dit_forward(noise, t_tensor, features)
        # MeanFlowerVLA uses dit_forward_meanflow(z, t, h, cond_dict) where h = t - r
        h_tensor = torch.ones_like(t_tensor)
        return model.dit_forward_meanflow(noise, t_tensor, h_tensor, features)

    @torch.no_grad()
    def fn_head():
        return model.sample_actions(noise.clone(), features, inference=True)

    @torch.no_grad()
    def fn_total():
        return model(obs, goal)

    LOG.info("Timing VLM (encode_observations) ...")
    vlm_ms = _cuda_time(fn_vlm, n_passes=n_passes, n_warmup=n_warmup)
    LOG.info("Timing DiT single step ...")
    dit_ms = _cuda_time(fn_dit_single, n_passes=n_passes, n_warmup=n_warmup)
    LOG.info("Timing head (sample_actions, num_sampling_steps=%d) ...", model.num_sampling_steps)
    head_ms = _cuda_time(fn_head, n_passes=n_passes, n_warmup=n_warmup)
    LOG.info("Timing total (model.forward) ...")
    total_ms = _cuda_time(fn_total, n_passes=n_passes, n_warmup=n_warmup)
    return {"vlm_ms": vlm_ms, "dit_ms": dit_ms, "head_ms": head_ms, "total_ms": total_ms}


def _hardware_tag() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0).replace(" ", "")
    cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
    dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return f"{name}-cap{cap}-{dtype}"


def _identify_variant(cfg: DictConfig) -> dict:
    """Lightweight variant ID for the JSON header (does not import the
    package: avoids a circular dep and lets bench_inference.py run without
    the analysis package installed)."""
    target = OmegaConf.select(cfg, "model._target_") or ""
    n_sampling = int(OmegaConf.select(cfg, "model.num_sampling_steps") or 0)
    use_imf = bool(OmegaConf.select(cfg, "model.use_imf") or False)
    ratio = OmegaConf.select(cfg, "model.ratio")
    head_depth = OmegaConf.select(cfg, "model.imf_head_depth")
    is_rf = target.endswith(".FlowerVLA") or "flower.models.flower." in target
    if is_rf:
        family = "rf"
        label = "rf" if n_sampling == 4 else f"rf_n{n_sampling}"
    elif "MeanFlowerVLA" in target or "meanflower" in target.lower():
        family = "imf" if use_imf else "mf_naive"
        if family == "mf_naive":
            label = "mf_naive"
        else:
            r = float(ratio) if ratio is not None else 0.25
            d = int(head_depth) if head_depth is not None else 8
            label = "imf" if (abs(r - 0.25) < 1e-6 and d == 8) else f"imf_r{r:.2f}_lh{d}"
    else:
        family = "unknown"
        label = target or "unknown"
    return {
        "variant": label,
        "family": family,
        "n_sampling": n_sampling,
        "ratio": float(ratio) if ratio is not None else None,
        "imf_head_depth": int(head_depth) if head_depth is not None else None,
    }


@hydra.main(config_path="../../conf", config_name="eval_libero", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    seed_everything(0, workers=True)

    bench_cfg = OmegaConf.select(cfg, "bench") or OmegaConf.create({})
    n_passes = int(OmegaConf.select(bench_cfg, "n_passes") or 200)
    n_warmup = int(OmegaConf.select(bench_cfg, "n_warmup") or 20)
    out_json = OmegaConf.select(bench_cfg, "out_json")
    if out_json is None:
        raise SystemExit("Provide bench.out_json=path/to/output.json")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for `torch.cuda.Event` timing.")

    LOG.info("Loading model: train_folder=%s, checkpoint=%s", cfg.train_folder, cfg.checkpoint)
    model, _, _, _ = get_default_mode_and_env(
        cfg.train_folder,
        cfg.dataset_path,
        cfg.checkpoint,
        env=42,
        lang_embeddings=None,
        eval_cfg_overwrite=cfg.eval_cfg_overwrite,
        device_id=cfg.device,
        prep_dm_and_deps=False,
    )
    device = torch.device(f"cuda:{cfg.device}" if isinstance(cfg.device, int) else cfg.device)
    model = model.to(device)
    model.eval()
    model.reset()

    obs, goal = _make_synthetic_obs(model, device)
    timings = _bench_components(model, obs, goal, n_passes=n_passes, n_warmup=n_warmup)

    # Resolve the train_folder Hydra config so we can read the *training*
    # config (not the eval override).  This is what variant ID needs.
    try:
        train_cfg = OmegaConf.load(cfg.train_folder)
        variant_info = _identify_variant(train_cfg)
    except Exception as e:
        LOG.warning("Could not load train_folder config (%s); using eval cfg for variant ID", e)
        variant_info = _identify_variant(cfg)

    # Allow `+variant_label=...` (same flag the eval script uses) to override
    # the auto-detected variant so latency JSONs match episodes.jsonl variants.
    override = OmegaConf.select(cfg, "variant_label")
    if override:
        variant_info["variant"] = str(override)

    payload = {
        **variant_info,
        "hardware": _hardware_tag(),
        "n_passes": n_passes,
        "n_warmup": n_warmup,
        "torch_version": torch.__version__,
        "vlm_ms": timings["vlm_ms"].tolist(),
        "dit_ms": timings["dit_ms"].tolist(),
        "head_ms": timings["head_ms"].tolist(),
        "total_ms": timings["total_ms"].tolist(),
        "summary": {
            comp: {
                "mean": float(timings[f"{comp}_ms"].mean()),
                "std": float(timings[f"{comp}_ms"].std()),
                "median": float(np.median(timings[f"{comp}_ms"])),
            }
            for comp in ("vlm", "dit", "head", "total")
        },
        "checkpoint": str(cfg.checkpoint),
        "train_folder": str(cfg.train_folder),
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    LOG.info(
        "Wrote %s | total %.2f ± %.2f ms | head %.2f ± %.2f ms | vlm %.2f ± %.2f ms",
        out_path,
        payload["summary"]["total"]["mean"], payload["summary"]["total"]["std"],
        payload["summary"]["head"]["mean"], payload["summary"]["head"]["std"],
        payload["summary"]["vlm"]["mean"], payload["summary"]["vlm"]["std"],
    )


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    main()
