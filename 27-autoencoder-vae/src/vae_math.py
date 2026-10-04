"""Numerically stable VAE mathematical primitives implemented with NumPy."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def reparameterize(
    mu: np.ndarray,
    logvar: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.asarray(mu, dtype=float)
    lv = np.asarray(logvar, dtype=float)
    if mean.shape != lv.shape:
        raise ValueError("mu and logvar must have equal shapes")
    epsilon = rng.normal(size=mean.shape)
    std = np.exp(0.5 * lv)
    return mean + std * epsilon


def kl_standard_normal(
    mu: np.ndarray,
    logvar: np.ndarray,
    *,
    reduction: str = "mean",
) -> float | np.ndarray:
    mean = np.asarray(mu, dtype=float)
    lv = np.asarray(logvar, dtype=float)
    if mean.shape != lv.shape:
        raise ValueError("mu and logvar must have equal shapes")
    if mean.ndim < 2:
        raise ValueError("expected batch dimension and latent dimension")

    per_sample = -0.5 * np.sum(
        1.0 + lv - mean**2 - np.exp(lv),
        axis=-1,
    )

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return float(np.sum(per_sample))
    if reduction == "mean":
        return float(np.mean(per_sample))
    raise ValueError("reduction must be none, sum, or mean")


def binary_cross_entropy_with_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    reduction: str = "mean",
) -> float | np.ndarray:
    z = np.asarray(logits, dtype=float)
    y = np.asarray(targets, dtype=float)

    if z.shape != y.shape:
        raise ValueError("logits and targets must have equal shapes")
    if not np.all((y >= 0.0) & (y <= 1.0)):
        raise ValueError("targets must lie in [0,1]")

    # Stable softplus(z) - y*z.
    loss = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))

    if reduction == "none":
        return loss
    if reduction == "sum":
        return float(np.sum(loss))
    if reduction == "mean":
        return float(np.mean(loss))
    raise ValueError("reduction must be none, sum, or mean")


def vae_loss(
    reconstruction_logits: np.ndarray,
    targets: np.ndarray,
    mu: np.ndarray,
    logvar: np.ndarray,
    *,
    beta: float = 1.0,
) -> dict[str, float]:
    if beta < 0:
        raise ValueError("beta must be non-negative")

    logits = np.asarray(reconstruction_logits, dtype=float)
    y = np.asarray(targets, dtype=float)

    if logits.ndim < 2:
        raise ValueError("expected batch dimension")

    recon_elementwise = binary_cross_entropy_with_logits(
        logits,
        y,
        reduction="none",
    )
    reconstruction = float(
        np.mean(np.sum(recon_elementwise.reshape(len(logits), -1), axis=1))
    )
    kl = float(kl_standard_normal(mu, logvar, reduction="mean"))
    total = reconstruction + beta * kl

    return {
        "total": total,
        "reconstruction": reconstruction,
        "kl": kl,
    }
