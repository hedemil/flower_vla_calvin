"""Statistical helpers for the Results chapter (`docs/thesis_results.md` §6.3).

All helpers operate on integer success counts or boolean per-episode arrays.
No matplotlib imports here — figure code lives in the entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np


# ----- Clopper-Pearson ------------------------------------------------------

@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float

    @property
    def lower(self) -> float:
        return self.lo

    @property
    def upper(self) -> float:
        return self.hi


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> CI:
    """Exact two-sided Clopper-Pearson interval on a binomial proportion.

    Uses ``scipy.stats.binomtest(...).proportion_ci(method="exact")``.
    Returns fractions in [0, 1]; convert to pp at print time.
    """
    if n == 0:
        return CI(point=float("nan"), lo=0.0, hi=1.0)
    if k < 0 or k > n:
        raise ValueError(f"k={k} not in [0, n={n}]")
    from scipy.stats import binomtest

    res = binomtest(k=int(k), n=int(n))
    interval = res.proportion_ci(confidence_level=1 - alpha, method="exact")
    return CI(point=k / n, lo=float(interval.low), hi=float(interval.high))


def average_ci(intervals: Sequence[CI]) -> CI:
    """Per-spec §2 'CI format': suite-averaged CI = mean of per-task CIs.

    This is *not* a CI on the suite mean — it's the mean of the bounds, used
    to convey the typical task-level uncertainty.
    """
    if not intervals:
        return CI(point=float("nan"), lo=float("nan"), hi=float("nan"))
    pts = np.array([c.point for c in intervals], dtype=float)
    los = np.array([c.lo for c in intervals], dtype=float)
    his = np.array([c.hi for c in intervals], dtype=float)
    return CI(point=float(np.nanmean(pts)), lo=float(np.nanmean(los)), hi=float(np.nanmean(his)))


# ----- McNemar --------------------------------------------------------------

def _to_bool_array(x: Iterable[int | bool]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=int)
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("outcomes must be 0/1 or boolean")
    return arr.astype(bool)


def mcnemar_table(outcomes_a: Iterable[int | bool], outcomes_b: Iterable[int | bool]) -> np.ndarray:
    """Build the 2x2 contingency table on paired binary outcomes.

    Rows = A success/fail; cols = B success/fail.
    """
    a = _to_bool_array(outcomes_a)
    b = _to_bool_array(outcomes_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    return np.array([[n11, n10], [n01, n00]], dtype=int)


def paired_mcnemar(
    outcomes_a: Iterable[int | bool],
    outcomes_b: Iterable[int | bool],
    *,
    exact: bool = True,
) -> float:
    """Return the McNemar p-value on paired binary outcomes (two-sided).

    Exact test: p = 2 * Binomial(min(b,c); b+c, 0.5) via scipy.
    Asymptotic fallback (exact=False): chi-squared with continuity correction.
    """
    from scipy.stats import binomtest, chi2

    table = mcnemar_table(outcomes_a, outcomes_b)
    b = int(table[0, 1])  # A success, B fail
    c = int(table[1, 0])  # A fail, B success
    n = b + c
    if n == 0:
        return 1.0
    if exact:
        return float(binomtest(min(b, c), n, 0.5).pvalue)
    # Edwards continuity correction
    stat = (abs(b - c) - 1) ** 2 / n
    return float(1.0 - chi2.cdf(stat, df=1))


def bonferroni_threshold(alpha: float, n_pairs: int) -> float:
    """Bonferroni-corrected significance threshold."""
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    return alpha / n_pairs


# ----- Paired difference CI (Fig. R5 whiskers) ------------------------------

def paired_diff_ci(
    outcomes_a: Iterable[int | bool],
    outcomes_b: Iterable[int | bool],
    *,
    alpha: float = 0.05,
) -> CI:
    """Wilson-style CI on the per-episode signed paired difference (a - b).

    For paired binary outcomes, only the discordant pairs (b=succ&a=fail vs.
    a=succ&b=fail) inform the difference.  Following Newcombe (1998), we
    construct the difference CI from the marginal Wilson intervals and the
    discordant counts.  Returns the difference in fractions, signed (a - b).
    """
    a = _to_bool_array(outcomes_a)
    b = _to_bool_array(outcomes_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n = int(a.size)
    if n == 0:
        return CI(point=0.0, lo=0.0, hi=0.0)

    p_a = float(np.mean(a))
    p_b = float(np.mean(b))
    diff = p_a - p_b

    from scipy.stats import norm as _norm
    z = _norm.ppf(1 - alpha / 2)

    def _wilson(k: int, _n: int) -> tuple[float, float]:
        p = k / _n
        d = 1 + z ** 2 / _n
        c = (p + z ** 2 / (2 * _n)) / d
        m = z * (p * (1 - p) / _n + z ** 2 / (4 * _n ** 2)) ** 0.5 / d
        return c - m, c + m

    lo_a, hi_a = _wilson(int(a.sum()), n)
    lo_b, hi_b = _wilson(int(b.sum()), n)

    # Newcombe Method 10: paired-difference CI using marginal Wilson bounds
    # plus the phi correction from discordant pairs.
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    denom = (n10 + n11) * (n01 + n11) * (n10 + n00) * (n01 + n00)
    if denom == 0:
        phi = 0.0
    else:
        phi = (n11 * n00 - n10 * n01) / float(denom) ** 0.5

    delta_lo = ((p_a - lo_a) ** 2 - 2 * phi * (p_a - lo_a) * (hi_b - p_b) + (hi_b - p_b) ** 2) ** 0.5
    delta_hi = ((hi_a - p_a) ** 2 - 2 * phi * (hi_a - p_a) * (p_b - lo_b) + (p_b - lo_b) ** 2) ** 0.5
    return CI(point=diff, lo=diff - delta_lo, hi=diff + delta_hi)


# ----- Verdicts -------------------------------------------------------------

Verdict = Literal["pass", "fail", "inconclusive"]


def non_inferiority_verdict(diff_point: float, diff_lo: float, delta: float) -> Verdict:
    """Per `thesis_results.md` §6 Table R6:

    - pass:           CI lower bound > -delta.
    - fail:           CI upper bound entirely below -delta (point estimate < -delta).
    - inconclusive:   CI straddles -delta but point estimate is non-negative.

    `delta` is in the same units as `diff_*` (fractions, not pp).
    """
    if diff_lo > -delta:
        return "pass"
    if diff_point < -delta:
        return "fail"
    return "inconclusive"


def latency_speedup_verdict(rf_mean: float, rf_std: float, var_mean: float, var_std: float) -> Verdict:
    """Per Table R7: `pass` if speed-up > 1.0× and the variant's
    (mean + 2σ) latency stays below the RF mean (i.e. clear separation)."""
    if var_mean >= rf_mean:
        return "fail"
    if (var_mean + 2 * var_std) < rf_mean:
        return "pass"
    return "inconclusive"


# ----- Stability detectors (Table R1) ---------------------------------------

def first_spike_step(
    steps: np.ndarray,
    values: np.ndarray,
    *,
    spike_factor: float = 1.5,
    rolling_window: int = 2000,
) -> int | None:
    """First step where ``values[t] > spike_factor * rolling_min(values, window)``.

    Returns None if no spike detected.  Used for `raw_mse` spike onset
    (Fig. R1 annotation, Table R1 column).
    """
    if values.size == 0:
        return None
    rolling_min = np.minimum.accumulate(values)  # cheap proxy
    # Window-based rolling min via cumulative trick:
    # use pandas if available for the proper sliding window.
    try:
        import pandas as pd
        s = pd.Series(values)
        rolling_min = s.rolling(window=rolling_window, min_periods=1).min().to_numpy()
    except ImportError:
        pass
    spike_mask = values > spike_factor * np.maximum(rolling_min, 1e-12)
    idxs = np.where(spike_mask)[0]
    if idxs.size == 0:
        return None
    return int(steps[idxs[0]])


def first_divergence_step(
    steps: np.ndarray,
    val_loss: np.ndarray,
    *,
    baseline_step: int = 10_000,
    factor: float = 1.5,
    n_consecutive: int = 5,
) -> int | None:
    """First step `s` such that `val_loss[s'] > factor * val_loss(baseline)`
    for `n_consecutive` consecutive validation evaluations starting at `s`.

    Per `thesis_results.md` §3 Table R1: 'val_loss(step) > 1.5 × val_loss(10k)
    for five consecutive validation evaluations'.
    """
    if val_loss.size == 0:
        return None
    base_idx = int(np.searchsorted(steps, baseline_step))
    if base_idx >= val_loss.size:
        return None
    baseline = float(val_loss[base_idx])
    threshold = factor * baseline
    # Only consider steps at or after baseline_step — a healthy training
    # curve typically has val_loss > baseline early on while still
    # converging.
    eligible = steps >= baseline_step
    over = (val_loss > threshold) & eligible
    if over.size < n_consecutive:
        return None
    win = np.lib.stride_tricks.sliding_window_view(over, n_consecutive)
    starts = np.where(np.all(win, axis=1))[0]
    if starts.size == 0:
        return None
    return int(steps[starts[0]])
