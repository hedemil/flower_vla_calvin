import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from safetensors.torch import save_file

from flower.callbacks.ema import EMA

logger = logging.getLogger(__name__)


class EMASafetensorsCheckpoint(Callback):
    """Save EMA-smoothed weights as safetensors when a monitored metric improves.

    Pairs with the EMA callback's `evaluate_ema_weights_instead=True` mode: at
    `on_validation_epoch_end` the live model still holds EMA weights, so
    `pl_module.state_dict()` IS the EMA state dict. We serialize that directly.

    Why a separate path from PL's ModelCheckpoint: enabling `save_weights_only=False`
    would also persist AdamW optimizer state (~2x model size) and OOMs on this setup.
    Safetensors is memory-mapped and tensor-only, sidestepping both issues.
    """

    def __init__(
        self,
        dirpath: str,
        monitor: str = "eval_lh/avg_seq_len",
        mode: str = "max",
        every_n_epochs: int = 10,
        hydra_config_path: Optional[str] = None,
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.every_n_epochs = every_n_epochs
        self._hydra_config_path = Path(hydra_config_path) if hydra_config_path else None
        self.best = float("-inf") if mode == "max" else float("inf")

    def _resolve_hydra_config(self) -> Optional[Path]:
        if self._hydra_config_path is not None and self._hydra_config_path.exists():
            return self._hydra_config_path
        try:
            from hydra.core.hydra_config import HydraConfig
            cfg = HydraConfig.get()
            candidate = Path(cfg.runtime.output_dir) / ".hydra" / "config.yaml"
            if candidate.exists():
                self._hydra_config_path = candidate
                return candidate
        except Exception as e:
            logger.warning(f"EMASafetensorsCheckpoint: could not resolve Hydra config: {e}")
        return None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if self.every_n_epochs > 0 and trainer.current_epoch % self.every_n_epochs != 0:
            return

        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        metric_val = metric.item() if isinstance(metric, torch.Tensor) else float(metric)

        improved = (metric_val > self.best) if self.mode == "max" else (metric_val < self.best)
        if not improved:
            return

        ema_cb = next((cb for cb in trainer.callbacks if isinstance(cb, EMA)), None)
        if ema_cb is None or not ema_cb.ema_initialized:
            logger.warning("EMASafetensorsCheckpoint: no initialized EMA callback found, skipping save")
            return

        self.best = metric_val

        # With evaluate_ema_weights_instead=True, EMA weights are swapped into the
        # live model at on_validation_start and restored at on_validation_end.
        # on_validation_epoch_end fires between those, so state_dict() is EMA.
        state_dict = {k: v.detach().cpu().contiguous() for k, v in pl_module.state_dict().items()}

        out_dir = self.dirpath / f"epoch={trainer.current_epoch:02d}_{self.monitor.replace('/', '_')}={metric_val:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(out_dir / "model.safetensors"))

        cfg_path = self._resolve_hydra_config()
        if cfg_path is not None:
            shutil.copy(str(cfg_path), str(out_dir / "config.yaml"))

        logger.info(f"EMASafetensorsCheckpoint: saved EMA weights to {out_dir} ({self.monitor}={metric_val:.4f})")
