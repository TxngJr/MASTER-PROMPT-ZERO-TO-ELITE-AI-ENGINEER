"""Mathematical primitives spanning major generative-model families."""

from __future__ import annotations

import numpy as np


def gaussian_nll(
    x: np.ndarray,
    mean: np.ndarray,
    log_variance: np.ndarray | float,
    *,
    reduction: str = "mean",
) -> float | np.ndarray:
    values = np.asarray(x, dtype=float)
    mu = np.asarray(mean, dtype=float)
    logvar = np.asarray(log_variance, dtype=float)

    try:
        mu = np.broadcast_to(mu, values.shape)
        logvar = np.broadcast_to(logvar, values.shape)
    except ValueError as exc:
        raise ValueError("mean/log_variance not broadcastable to x") from exc

    loss = 0.5 * (
        np.log(2.0 * np.pi)
        + logvar
        + (values - mu) ** 2 / np.exp(logvar)
    )

    if reduction == "none":
        return loss
    if reduction == "sum":
        return float(np.sum(loss))
    if reduction == "mean":
        return float(np.mean(loss))
    raise ValueError("reduction must be none, sum, or mean")


def categorical_nll(
    logits: np.ndarray,
    target_index: int,
) -> float:
    values = np.asarray(logits, dtype=float).reshape(-1)

    if not 0 <= target_index < len(values):
        raise ValueError("target_index out of range")

    maximum = np.max(values)
    logsumexp = maximum + np.log(
        np.sum(np.exp(values - maximum))
    )
    return float(logsumexp - values[target_index])


def forward_diffusion_sample(
    x0: np.ndarray,
    alpha_bar: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= alpha_bar <= 1.0:
        raise ValueError("alpha_bar must lie in [0,1]")

    clean = np.asarray(x0, dtype=float)
    noise = rng.normal(size=clean.shape)
    noisy = (
        np.sqrt(alpha_bar) * clean
        + np.sqrt(1.0 - alpha_bar) * noise
    )
    return noisy, noise


def energy_to_probability(energies: np.ndarray) -> np.ndarray:
    e = np.asarray(energies, dtype=float).reshape(-1)
    logits = -e
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def effective_sample_size(
    normalized_weights: np.ndarray,
) -> float:
    weights = np.asarray(normalized_weights, dtype=float).reshape(-1)

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must have positive total")

    weights = weights / total
    return float(1.0 / np.sum(weights**2))
