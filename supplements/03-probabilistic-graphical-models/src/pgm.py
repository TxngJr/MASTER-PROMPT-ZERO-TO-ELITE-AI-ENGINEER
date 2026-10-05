from __future__ import annotations
import numpy as np


def normalize(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("p must be a finite non-negative vector")
    s = p.sum()
    if s <= 0:
        raise ValueError("probability mass must be positive")
    return p / s


def bayes_update(prior, likelihood):
    prior = normalize(prior)
    likelihood = np.asarray(likelihood, dtype=float)
    if likelihood.shape != prior.shape or np.any(likelihood < 0):
        raise ValueError("likelihood must align and be non-negative")
    return normalize(prior * likelihood)


def hmm_forward(initial, transition, emissions):
    initial = normalize(initial)
    transition = np.asarray(transition, dtype=float)
    emissions = np.asarray(emissions, dtype=float)
    k = initial.shape[0]
    if transition.shape != (k, k):
        raise ValueError("transition shape mismatch")
    if emissions.ndim != 2 or emissions.shape[1] != k:
        raise ValueError("emissions must have shape (time, states)")
    if np.any(transition < 0) or np.any(emissions < 0):
        raise ValueError("probabilities must be non-negative")
    row_sums = transition.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("transition rows must sum to 1")
    alpha = initial * emissions[0]
    history = [alpha.copy()]
    for t in range(1, len(emissions)):
        alpha = (alpha @ transition) * emissions[t]
        history.append(alpha.copy())
    return np.asarray(history), float(alpha.sum())
