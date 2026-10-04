"""Numerically stable GAN objective primitives implemented with NumPy."""

from __future__ import annotations

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))


def bce_with_logits(
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

    loss = softplus(z) - z * y

    if reduction == "none":
        return loss
    if reduction == "sum":
        return float(np.sum(loss))
    if reduction == "mean":
        return float(np.mean(loss))
    raise ValueError("reduction must be none, sum, or mean")


def discriminator_loss(
    real_logits: np.ndarray,
    fake_logits: np.ndarray,
) -> float:
    real = np.asarray(real_logits, dtype=float)
    fake = np.asarray(fake_logits, dtype=float)

    real_loss = bce_with_logits(real, np.ones_like(real))
    fake_loss = bce_with_logits(fake, np.zeros_like(fake))

    return 0.5 * (float(real_loss) + float(fake_loss))


def generator_non_saturating_loss(fake_logits: np.ndarray) -> float:
    fake = np.asarray(fake_logits, dtype=float)
    return float(bce_with_logits(fake, np.ones_like(fake)))


def generator_saturating_loss(fake_logits: np.ndarray) -> float:
    """Original minimax G objective E[log(1-sigmoid(logit))]."""
    fake = np.asarray(fake_logits, dtype=float)
    return float(np.mean(-softplus(fake)))
