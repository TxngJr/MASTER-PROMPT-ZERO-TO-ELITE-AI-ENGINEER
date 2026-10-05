from __future__ import annotations
import numpy as np


def difference_in_means(y, treatment) -> float:
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    if y.shape != t.shape or not np.any(t == 1) or not np.any(t == 0):
        raise ValueError("need aligned outcomes with treated and control observations")
    return float(y[t == 1].mean() - y[t == 0].mean())


def standardized_mean_difference(x, treatment) -> float:
    x = np.asarray(x, dtype=float)
    t = np.asarray(treatment, dtype=int)
    a, b = x[t == 1], x[t == 0]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("need at least two observations per group")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0 if a.mean() == b.mean() else float("inf")
    return float((a.mean() - b.mean()) / pooled)


def ipw_ate(y, treatment, propensity, clip: float = 1e-3) -> float:
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=float)
    p = np.asarray(propensity, dtype=float)
    if not (y.shape == t.shape == p.shape):
        raise ValueError("arrays must align")
    if not (0 < clip < 0.5):
        raise ValueError("clip must be between 0 and 0.5")
    p = np.clip(p, clip, 1 - clip)
    return float(np.mean(t * y / p - (1 - t) * y / (1 - p)))
