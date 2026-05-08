"""Data-driven variant identification from W&B run configs.

Per the design discussion: do not hardcode the iMF ablation set against
``docs/thesis_results.md``.  Whatever ``(model._target_, ratio,
imf_head_depth, num_sampling_steps, use_imf)`` tuple is observed in the
W&B run history defines a variant.  Defaults are mapped to canonical
labels (``rf``, ``mf_naive``, ``imf``); off-defaults get parametric
labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


# Defaults from `conf/model/{flower,meanflower,imf}.yaml`.  Used to decide
# when an iMF run gets the canonical `imf` label vs. a parametric one.
IMF_DEFAULT_RATIO = 0.25
IMF_DEFAULT_HEAD_DEPTH = 8
RF_DEFAULT_N_SAMPLING = 4
MF_DEFAULT_N_SAMPLING = 1
IMF_DEFAULT_N_SAMPLING = 1


@dataclass(frozen=True)
class VariantKey:
    """Stable identifier for a single (variant) cell."""

    family: str  # "rf" | "mf_naive" | "imf"
    n_sampling: int | None = None
    ratio: float | None = None
    head_depth: int | None = None
    use_imf: bool | None = None

    def label(self) -> str:
        """Canonical-or-parametric label string used as a column key."""
        if self.family == "rf":
            if self.n_sampling == RF_DEFAULT_N_SAMPLING:
                return "rf"
            return f"rf_n{self.n_sampling}"
        if self.family == "mf_naive":
            return "mf_naive"
        # iMF family
        ratio = self.ratio if self.ratio is not None else IMF_DEFAULT_RATIO
        depth = self.head_depth if self.head_depth is not None else IMF_DEFAULT_HEAD_DEPTH
        is_default = (
            abs(ratio - IMF_DEFAULT_RATIO) < 1e-6
            and depth == IMF_DEFAULT_HEAD_DEPTH
        )
        if is_default:
            return "imf"
        return f"imf_r{ratio:.2f}_lh{depth}"


def _get(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Safe nested get: ``_get(cfg, 'model', 'ratio')``.

    W&B's CSV export sometimes flattens config keys with dots
    (``model.ratio``) and sometimes preserves nesting; tolerate either.
    """
    cur: Any = cfg
    for key in keys:
        if isinstance(cur, Mapping):
            if key in cur:
                cur = cur[key]
                continue
        # Try dotted form against the original mapping
        dotted = ".".join(keys)
        if isinstance(cfg, Mapping) and dotted in cfg:
            return cfg[dotted]
        return default
    return cur


def _get_first(cfg: Mapping[str, Any], paths: list[tuple[str, ...]], default: Any = None) -> Any:
    for path in paths:
        val = _get(cfg, *path, default=None)
        if val is not None:
            return val
    return default


def identify_variant(config: Mapping[str, Any]) -> VariantKey:
    """Map a W&B run config dict to a :class:`VariantKey`.

    Tolerant to:
    - Hydra-style nested configs (``{"model": {"_target_": "...", "ratio": 0.25}}``).
    - W&B-flattened configs (``{"model._target_": "...", "model.ratio": 0.25}``).
    """
    target = _get_first(
        config,
        [("model", "_target_"), ("model._target_",)],
        default="",
    )
    target = str(target or "")

    # Match the RF class strictly: dotted path .FlowerVLA at the end (so we
    # don't catch MeanFlowerVLA via a substring), or the canonical module path.
    is_rf = target.endswith(".FlowerVLA") or "flower.models.flower." in target
    if is_rf:
        n_sampling = int(
            _get_first(
                config,
                [("model", "num_sampling_steps"), ("model.num_sampling_steps",)],
                default=RF_DEFAULT_N_SAMPLING,
            )
        )
        return VariantKey(family="rf", n_sampling=n_sampling)

    if "meanflower" in target.lower() or "MeanFlowerVLA" in target:
        use_imf = bool(
            _get_first(
                config,
                [("model", "use_imf"), ("model.use_imf",)],
                default=False,
            )
        )
        n_sampling = int(
            _get_first(
                config,
                [("model", "num_sampling_steps"), ("model.num_sampling_steps",)],
                default=MF_DEFAULT_N_SAMPLING,
            )
        )
        if not use_imf:
            return VariantKey(family="mf_naive", n_sampling=n_sampling, use_imf=False)
        ratio = float(
            _get_first(
                config,
                [("model", "ratio"), ("model.ratio",)],
                default=IMF_DEFAULT_RATIO,
            )
        )
        depth = int(
            _get_first(
                config,
                [("model", "imf_head_depth"), ("model.imf_head_depth",)],
                default=IMF_DEFAULT_HEAD_DEPTH,
            )
        )
        return VariantKey(
            family="imf",
            n_sampling=n_sampling,
            ratio=ratio,
            head_depth=depth,
            use_imf=True,
        )

    # Unknown family — fall back on a parametric label keyed on _target_.
    return VariantKey(family=f"unknown:{target}")


def discover_variants(configs: Iterable[Mapping[str, Any]]) -> dict[str, VariantKey]:
    """Return ``{label: VariantKey}`` for every distinct variant observed.

    Used by entrypoints to build the column set of Tables R2/R3/R5 from the
    actual run history rather than the spec's hardcoded list.
    """
    out: dict[str, VariantKey] = {}
    for cfg in configs:
        key = identify_variant(cfg)
        out.setdefault(key.label(), key)
    return out
