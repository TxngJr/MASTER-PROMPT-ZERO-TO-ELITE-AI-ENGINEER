from __future__ import annotations
import numpy as np


def lif_step(voltage, current, beta: float = 0.9, threshold: float = 1.0, reset: float = 0.0):
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0,1)")
    if threshold <= reset:
        raise ValueError("threshold must exceed reset")
    v = beta * np.asarray(voltage, dtype=float) + np.asarray(current, dtype=float)
    spike = v >= threshold
    v = np.where(spike, reset, v)
    return v, spike.astype(np.int8)


def simulate_lif(currents, beta: float = 0.9, threshold: float = 1.0, reset: float = 0.0):
    x = np.asarray(currents, dtype=float)
    if x.ndim != 2:
        raise ValueError("currents must have shape (time, neurons)")
    v = np.zeros(x.shape[1], dtype=float)
    voltages, spikes = [], []
    for current in x:
        v, s = lif_step(v, current, beta, threshold, reset)
        voltages.append(v.copy())
        spikes.append(s)
    return np.asarray(voltages), np.asarray(spikes)


def spike_rate(spikes) -> np.ndarray:
    s = np.asarray(spikes)
    if s.ndim != 2 or len(s) == 0:
        raise ValueError("spikes must have shape (time, neurons)")
    if not np.all((s == 0) | (s == 1)):
        raise ValueError("spikes must be binary")
    return s.mean(axis=0)
