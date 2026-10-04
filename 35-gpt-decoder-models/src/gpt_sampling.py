"""GPT-style causal LM data and sampling utilities implemented with NumPy."""

from __future__ import annotations

import numpy as np


def make_causal_lm_pairs(
    token_ids: list[int],
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if len(token_ids) <= context_length:
        raise ValueError("token sequence too short")

    inputs = []
    targets = []

    for start in range(len(token_ids) - context_length):
        chunk = token_ids[start : start + context_length + 1]
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    return (
        np.asarray(inputs, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    )


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def apply_temperature(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return np.asarray(logits, dtype=float) / temperature


def top_k_filter(
    logits: np.ndarray,
    k: int,
) -> np.ndarray:
    values = np.asarray(logits, dtype=float).copy()

    if values.ndim != 1:
        raise ValueError("top_k_filter expects 1D logits")
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= len(values):
        return values

    keep_indices = np.argpartition(values, -k)[-k:]
    mask = np.ones(len(values), dtype=bool)
    mask[keep_indices] = False
    values[mask] = -np.inf
    return values


def top_p_filter(
    logits: np.ndarray,
    p: float,
) -> np.ndarray:
    values = np.asarray(logits, dtype=float).copy()

    if values.ndim != 1:
        raise ValueError("top_p_filter expects 1D logits")
    if not 0.0 < p <= 1.0:
        raise ValueError("p must lie in (0,1]")
    if p == 1.0:
        return values

    order = np.argsort(values)[::-1]
    sorted_logits = values[order]
    sorted_probs = stable_softmax(sorted_logits)
    cumulative = np.cumsum(sorted_probs)

    # Keep the first token that crosses the threshold.
    remove = cumulative > p
    if len(remove) > 1:
        remove[1:] = remove[:-1]
    remove[0] = False

    filtered_sorted = sorted_logits.copy()
    filtered_sorted[remove] = -np.inf

    filtered = np.full_like(values, -np.inf)
    filtered[order] = filtered_sorted
    return filtered


def sample_from_logits(
    logits: np.ndarray,
    *,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    filtered = apply_temperature(logits, temperature)

    if top_k is not None:
        filtered = top_k_filter(filtered, top_k)

    if top_p is not None:
        filtered = top_p_filter(filtered, top_p)

    probabilities = stable_softmax(filtered)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("invalid probability distribution")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities do not sum to one")

    return int(rng.choice(len(probabilities), p=probabilities))
