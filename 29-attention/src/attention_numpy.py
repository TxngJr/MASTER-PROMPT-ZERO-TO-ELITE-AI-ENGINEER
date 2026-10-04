"""Scaled dot-product and multi-head self-attention implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.sum(exp_values, axis=axis, keepdims=True)
    return exp_values / denom


def causal_mask(query_length: int, key_length: int | None = None) -> np.ndarray:
    if query_length <= 0:
        raise ValueError("query_length must be positive")
    if key_length is None:
        key_length = query_length
    if key_length <= 0:
        raise ValueError("key_length must be positive")

    q_index = np.arange(query_length)[:, None]
    k_index = np.arange(key_length)[None, :]
    return k_index <= q_index


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    keep_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=float)
    k = np.asarray(key, dtype=float)
    v = np.asarray(value, dtype=float)

    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("Q/K/V must have at least 2 dimensions")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("batch/head prefixes must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query/key dimensions must match")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("key/value sequence lengths must match")

    scores = q @ np.swapaxes(k, -1, -2)
    scores = scores / np.sqrt(q.shape[-1])

    if keep_mask is not None:
        mask = np.asarray(keep_mask, dtype=bool)
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError("mask is not broadcastable to attention scores") from exc

        valid_rows = np.any(mask, axis=-1)
        if not np.all(valid_rows):
            raise ValueError("every query row must keep at least one key")

        scores = np.where(mask, scores, -np.inf)

    weights = stable_softmax(scores, axis=-1)
    output = weights @ v
    return output, weights


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 3:
        raise ValueError("expected shape (B,T,D)")
    if num_heads <= 0 or values.shape[-1] % num_heads != 0:
        raise ValueError("D must be divisible by num_heads")

    batch, time, model_dim = values.shape
    head_dim = model_dim // num_heads

    return (
        values.reshape(batch, time, num_heads, head_dim)
        .transpose(0, 2, 1, 3)
    )


def combine_heads(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 4:
        raise ValueError("expected shape (B,H,T,Dh)")

    batch, heads, time, head_dim = values.shape
    return (
        values.transpose(0, 2, 1, 3)
        .reshape(batch, time, heads * head_dim)
    )


@dataclass
class MultiHeadSelfAttentionNumPy:
    model_dim: int
    num_heads: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.model_dim <= 0 or self.num_heads <= 0:
            raise ValueError("dimensions must be positive")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.model_dim)

        self.w_q = rng.normal(0.0, scale, (self.model_dim, self.model_dim))
        self.w_k = rng.normal(0.0, scale, (self.model_dim, self.model_dim))
        self.w_v = rng.normal(0.0, scale, (self.model_dim, self.model_dim))
        self.w_o = rng.normal(0.0, scale, (self.model_dim, self.model_dim))

    def forward(
        self,
        x: np.ndarray,
        *,
        causal: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(x, dtype=float)
        if values.ndim != 3 or values.shape[-1] != self.model_dim:
            raise ValueError("expected shape (B,T,model_dim)")

        q = split_heads(values @ self.w_q, self.num_heads)
        k = split_heads(values @ self.w_k, self.num_heads)
        v = split_heads(values @ self.w_v, self.num_heads)

        mask = None
        if causal:
            time = values.shape[1]
            mask = causal_mask(time)[None, None, :, :]

        heads, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            keep_mask=mask,
        )
        output = combine_heads(heads) @ self.w_o
        return output, weights
