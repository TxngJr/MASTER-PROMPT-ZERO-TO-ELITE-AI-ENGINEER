from __future__ import annotations
import numpy as np


def finite_difference_derivative(values, dt: float):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 3 or dt <= 0:
        raise ValueError("need 1D values of length >=3 and positive dt")
    return (y[2:] - y[:-2]) / (2.0 * dt)


def exponential_decay_residual(times, values, rate: float):
    t = np.asarray(times, dtype=float)
    u = np.asarray(values, dtype=float)
    if t.ndim != 1 or u.shape != t.shape or len(t) < 3:
        raise ValueError("times and values must be aligned 1D arrays")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0]) or dt[0] <= 0:
        raise ValueError("this educational helper expects equally spaced times")
    du = finite_difference_derivative(u, float(dt[0]))
    return du + rate * u[1:-1]


def relative_l2(prediction, reference, eps: float = 1e-12) -> float:
    p = np.asarray(prediction, dtype=float)
    r = np.asarray(reference, dtype=float)
    if p.shape != r.shape:
        raise ValueError("shape mismatch")
    denom = max(float(np.linalg.norm(r)), eps)
    return float(np.linalg.norm(p - r) / denom)
