"""Educational tanh RNN forward pass implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_sequence(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 3:
        raise ValueError("expected batch-first sequence shape (B,T,D)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("sequence must be finite")
    return arr


@dataclass
class TanhRNN:
    input_size: int
    hidden_size: int
    output_size: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        if self.output_size is not None and self.output_size <= 0:
            raise ValueError("output_size must be positive")

        rng = np.random.default_rng(self.seed)
        self.w_xh = rng.normal(
            0.0,
            1.0 / np.sqrt(self.input_size),
            size=(self.input_size, self.hidden_size),
        )
        self.w_hh = rng.normal(
            0.0,
            1.0 / np.sqrt(self.hidden_size),
            size=(self.hidden_size, self.hidden_size),
        )
        self.b_h = np.zeros(self.hidden_size, dtype=float)

        if self.output_size is not None:
            self.w_hy = rng.normal(
                0.0,
                1.0 / np.sqrt(self.hidden_size),
                size=(self.hidden_size, self.output_size),
            )
            self.b_y = np.zeros(self.output_size, dtype=float)
        else:
            self.w_hy = None
            self.b_y = None

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        x = np.asarray(x_t, dtype=float)
        h = np.asarray(h_prev, dtype=float)

        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError("x_t must have shape (B,input_size)")
        if h.shape != (x.shape[0], self.hidden_size):
            raise ValueError("h_prev shape mismatch")

        return np.tanh(x @ self.w_xh + h @ self.w_hh + self.b_h)

    def forward(
        self,
        x: np.ndarray,
        h0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sequence = _validate_sequence(x)
        batch, timesteps, features = sequence.shape

        if features != self.input_size:
            raise ValueError("input feature size mismatch")

        if h0 is None:
            hidden = np.zeros((batch, self.hidden_size), dtype=float)
        else:
            hidden = np.asarray(h0, dtype=float).copy()
            if hidden.shape != (batch, self.hidden_size):
                raise ValueError("h0 shape mismatch")

        outputs = np.empty(
            (batch, timesteps, self.hidden_size),
            dtype=float,
        )

        for t in range(timesteps):
            hidden = self.step(sequence[:, t, :], hidden)
            outputs[:, t, :] = hidden

        return outputs, hidden

    def readout(self, hidden_sequence: np.ndarray) -> np.ndarray:
        if self.w_hy is None or self.b_y is None:
            raise RuntimeError("output_size was not configured")
        h = np.asarray(hidden_sequence, dtype=float)
        if h.ndim != 3 or h.shape[2] != self.hidden_size:
            raise ValueError("hidden_sequence shape mismatch")
        return h @ self.w_hy + self.b_y
