"""Educational DDPM forward/reverse math implemented with NumPy."""

from __future__ import annotations

import numpy as np


def linear_beta_schedule(
    timesteps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> np.ndarray:
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("require 0 < beta_start < beta_end < 1")
    return np.linspace(beta_start, beta_end, timesteps, dtype=float)


def alpha_terms(
    betas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(betas, dtype=float).reshape(-1)
    if len(b) == 0 or np.any((b <= 0.0) | (b >= 1.0)):
        raise ValueError("betas must lie in (0,1)")
    alphas = 1.0 - b
    alpha_bars = np.cumprod(alphas)
    return alphas, alpha_bars


def q_sample(
    x0: np.ndarray,
    alpha_bar_t: float,
    *,
    rng: np.random.Generator,
    noise: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= alpha_bar_t <= 1.0:
        raise ValueError("alpha_bar_t must lie in [0,1]")

    clean = np.asarray(x0, dtype=float)
    eps = (
        rng.normal(size=clean.shape)
        if noise is None
        else np.asarray(noise, dtype=float)
    )
    if eps.shape != clean.shape:
        raise ValueError("noise shape mismatch")

    noisy = (
        np.sqrt(alpha_bar_t) * clean
        + np.sqrt(1.0 - alpha_bar_t) * eps
    )
    return noisy, eps


def predict_x0_from_epsilon(
    xt: np.ndarray,
    epsilon_hat: np.ndarray,
    alpha_bar_t: float,
) -> np.ndarray:
    if not 0.0 < alpha_bar_t <= 1.0:
        raise ValueError("alpha_bar_t must lie in (0,1]")

    x = np.asarray(xt, dtype=float)
    eps = np.asarray(epsilon_hat, dtype=float)
    if x.shape != eps.shape:
        raise ValueError("shape mismatch")

    return (
        x - np.sqrt(1.0 - alpha_bar_t) * eps
    ) / np.sqrt(alpha_bar_t)


def posterior_variance(
    beta_t: float,
    alpha_bar_t: float,
    alpha_bar_previous: float,
) -> float:
    if not 0.0 < beta_t < 1.0:
        raise ValueError("beta_t must lie in (0,1)")
    if not 0.0 < alpha_bar_t <= 1.0:
        raise ValueError("alpha_bar_t must lie in (0,1]")
    if not 0.0 < alpha_bar_previous <= 1.0:
        raise ValueError("alpha_bar_previous must lie in (0,1]")

    return float(
        beta_t
        * (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar_t)
    )


def ddpm_reverse_mean(
    xt: np.ndarray,
    epsilon_hat: np.ndarray,
    *,
    beta_t: float,
    alpha_t: float,
    alpha_bar_t: float,
) -> np.ndarray:
    if not 0.0 < beta_t < 1.0:
        raise ValueError("beta_t must lie in (0,1)")
    if not 0.0 < alpha_t <= 1.0:
        raise ValueError("alpha_t must lie in (0,1]")
    if not 0.0 < alpha_bar_t < 1.0:
        raise ValueError("alpha_bar_t must lie in (0,1)")

    x = np.asarray(xt, dtype=float)
    eps = np.asarray(epsilon_hat, dtype=float)
    if x.shape != eps.shape:
        raise ValueError("shape mismatch")

    return (
        x
        - beta_t
        / np.sqrt(1.0 - alpha_bar_t)
        * eps
    ) / np.sqrt(alpha_t)
