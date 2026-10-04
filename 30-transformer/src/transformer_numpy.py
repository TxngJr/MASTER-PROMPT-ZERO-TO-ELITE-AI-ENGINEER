"""Educational Transformer block primitives implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def sinusoidal_position_encoding(
    length: int,
    model_dim: int,
) -> np.ndarray:
    if length <= 0 or model_dim <= 0:
        raise ValueError("length and model_dim must be positive")

    positions = np.arange(length, dtype=float)[:, None]
    pair_indices = np.arange(0, model_dim, 2, dtype=float)
    frequencies = np.exp(
        -np.log(10000.0) * pair_indices / model_dim
    )

    encoding = np.zeros((length, model_dim), dtype=float)
    encoding[:, 0::2] = np.sin(positions * frequencies)

    odd_count = encoding[:, 1::2].shape[1]
    if odd_count:
        encoding[:, 1::2] = np.cos(
            positions * frequencies[:odd_count]
        )

    return encoding


def layer_norm(
    x: np.ndarray,
    *,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if eps <= 0:
        raise ValueError("eps must be positive")

    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (values - mean) / np.sqrt(variance + eps)

    if gamma is not None:
        g = np.asarray(gamma, dtype=float)
        if g.shape != (values.shape[-1],):
            raise ValueError("gamma shape mismatch")
        normalized = normalized * g

    if beta is not None:
        b = np.asarray(beta, dtype=float)
        if b.shape != (values.shape[-1],):
            raise ValueError("beta shape mismatch")
        normalized = normalized + b

    return normalized


def gelu(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return 0.5 * values * (
        1.0
        + np.tanh(
            np.sqrt(2.0 / np.pi)
            * (values + 0.044715 * values**3)
        )
    )


def _split_heads(x: np.ndarray, heads: int) -> np.ndarray:
    batch, time, model_dim = x.shape
    if model_dim % heads != 0:
        raise ValueError("model_dim must be divisible by heads")
    head_dim = model_dim // heads
    return x.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)


def _combine_heads(x: np.ndarray) -> np.ndarray:
    batch, heads, time, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(
        batch,
        time,
        heads * head_dim,
    )


@dataclass
class TransformerBlockNumPy:
    model_dim: int
    num_heads: int
    ff_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.model_dim, self.num_heads, self.ff_dim) <= 0:
            raise ValueError("dimensions must be positive")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.model_dim)

        self.w_q = rng.normal(0.0, scale, (self.model_dim, self.model_dim))
        self.w_k = rng.normal(0.0, scale, (self.model_dim, self.model_dim))
        self.w_v = rng.normal(0.0, scale, (self.model_dim, self.model_dim))
        self.w_o = rng.normal(0.0, scale, (self.model_dim, self.model_dim))

        self.w1 = rng.normal(0.0, scale, (self.model_dim, self.ff_dim))
        self.b1 = np.zeros(self.ff_dim)
        self.w2 = rng.normal(
            0.0,
            1.0 / np.sqrt(self.ff_dim),
            (self.ff_dim, self.model_dim),
        )
        self.b2 = np.zeros(self.model_dim)

        self.norm1_gamma = np.ones(self.model_dim)
        self.norm1_beta = np.zeros(self.model_dim)
        self.norm2_gamma = np.ones(self.model_dim)
        self.norm2_beta = np.zeros(self.model_dim)

    def _causal_attention(self, x: np.ndarray) -> np.ndarray:
        batch, time, _ = x.shape
        q = _split_heads(x @ self.w_q, self.num_heads)
        k = _split_heads(x @ self.w_k, self.num_heads)
        v = _split_heads(x @ self.w_v, self.num_heads)

        head_dim = q.shape[-1]
        scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)

        keep = (
            np.arange(time)[None, :]
            <= np.arange(time)[:, None]
        )
        scores = np.where(
            keep[None, None, :, :],
            scores,
            -np.inf,
        )

        weights = stable_softmax(scores, axis=-1)
        attended = weights @ v
        return _combine_heads(attended) @ self.w_o

    def forward(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim != 3 or values.shape[-1] != self.model_dim:
            raise ValueError("expected shape (B,T,model_dim)")

        # Pre-norm decoder-style block.
        normed = layer_norm(
            values,
            gamma=self.norm1_gamma,
            beta=self.norm1_beta,
        )
        x_after_attn = values + self._causal_attention(normed)

        normed_ff = layer_norm(
            x_after_attn,
            gamma=self.norm2_gamma,
            beta=self.norm2_beta,
        )
        hidden = gelu(normed_ff @ self.w1 + self.b1)
        ff = hidden @ self.w2 + self.b2
        return x_after_attn + ff


def add_token_and_position_embeddings(
    token_embeddings: np.ndarray,
) -> np.ndarray:
    x = np.asarray(token_embeddings, dtype=float)
    if x.ndim != 3:
        raise ValueError("expected token embeddings with shape (B,T,D)")
    position = sinusoidal_position_encoding(x.shape[1], x.shape[2])
    return x + position[None, :, :]
