from __future__ import annotations

import numpy as np


def normalized_retention(before: float, after_injection: float, after_washout: float, eps: float = 1e-12) -> float:
    """Normalized retained behavioural change.

    Retention = (after_washout - before) / (after_injection - before)

    Values near 1 mean most of the injected change survived; values near 0 mean it washed out.
    Values can be negative if washout reverses the injected behaviour, or above 1 if the behaviour
    improves further during washout.
    """
    denom = after_injection - before
    if abs(denom) < eps:
        return float("nan")
    return float((after_washout - before) / denom)


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Nonparametric bootstrap CI for the mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = np.mean(sample)
    lo = (1.0 - ci) / 2.0
    hi = 1.0 - lo
    return float(np.quantile(means, lo)), float(np.quantile(means, hi))
