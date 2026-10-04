"""Educational multimodal contrastive and cross-attention primitives."""

from __future__ import annotations

import numpy as np


def l2_normalize(
    x: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norm, eps)


def similarity_logits(
    left: np.ndarray,
    right: np.ndarray,
    *,
    temperature: float = 0.07,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("embeddings must be matrices")
    if a.shape[1] != b.shape[1]:
        raise ValueError("embedding dimensions must match")

    a = l2_normalize(a)
    b = l2_normalize(b)
    return (a @ b.T) / temperature


def stable_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.asarray(logits, dtype=float)
    y = np.asarray(targets, dtype=np.int64)

    if scores.ndim != 2 or y.shape != (scores.shape[0],):
        raise ValueError("shape mismatch")
    if np.any((y < 0) | (y >= scores.shape[1])):
        raise ValueError("target out of range")

    maximum = np.max(scores, axis=1, keepdims=True)
    logsumexp = (
        maximum[:, 0]
        + np.log(np.sum(np.exp(scores - maximum), axis=1))
    )
    correct = scores[np.arange(len(y)), y]
    return float(np.mean(logsumexp - correct))


def symmetric_contrastive_loss(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    temperature: float = 0.07,
) -> float:
    images = np.asarray(image_embeddings, dtype=float)
    texts = np.asarray(text_embeddings, dtype=float)

    if images.shape != texts.shape:
        raise ValueError("paired embedding matrices must have equal shape")

    logits = similarity_logits(
        images,
        texts,
        temperature=temperature,
    )
    targets = np.arange(len(images), dtype=np.int64)

    return 0.5 * (
        stable_cross_entropy(logits, targets)
        + stable_cross_entropy(logits.T, targets)
    )


def retrieval_top1_accuracy(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> float:
    q = np.asarray(query_embeddings, dtype=float)
    c = np.asarray(candidate_embeddings, dtype=float)

    if q.shape != c.shape:
        raise ValueError("paired matrices must have equal shape")

    logits = similarity_logits(q, c, temperature=1.0)
    predicted = np.argmax(logits, axis=1)
    expected = np.arange(len(q))
    return float(np.mean(predicted == expected))


def simple_cross_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=float)
    k = np.asarray(key, dtype=float)
    v = np.asarray(value, dtype=float)

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("expected batched Q/K/V tensors")
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("batch sizes must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query/key dimensions must match")
    if k.shape[1] != v.shape[1]:
        raise ValueError("key/value sequence lengths must match")

    scores = q @ np.swapaxes(k, -1, -2)
    scores = scores / np.sqrt(q.shape[-1])
    scores = scores - np.max(scores, axis=-1, keepdims=True)

    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights @ v, weights
