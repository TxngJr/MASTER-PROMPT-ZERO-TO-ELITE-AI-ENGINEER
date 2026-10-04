"""Educational LSTM and GRU cells implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


@dataclass
class LSTMCellNumPy:
    input_size: int
    hidden_size: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("sizes must be positive")

        rng = np.random.default_rng(self.seed)
        scale_x = 1.0 / np.sqrt(self.input_size)
        scale_h = 1.0 / np.sqrt(self.hidden_size)

        self.w_x = rng.normal(
            0.0,
            scale_x,
            size=(self.input_size, 4 * self.hidden_size),
        )
        self.w_h = rng.normal(
            0.0,
            scale_h,
            size=(self.hidden_size, 4 * self.hidden_size),
        )
        self.bias = np.zeros(4 * self.hidden_size, dtype=float)

    def step(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        x = np.asarray(x_t, dtype=float)
        h = np.asarray(h_prev, dtype=float)
        c = np.asarray(c_prev, dtype=float)

        batch = x.shape[0]
        if x.shape != (batch, self.input_size):
            raise ValueError("x_t shape mismatch")
        if h.shape != (batch, self.hidden_size):
            raise ValueError("h_prev shape mismatch")
        if c.shape != (batch, self.hidden_size):
            raise ValueError("c_prev shape mismatch")

        gates = x @ self.w_x + h @ self.w_h + self.bias
        f_raw, i_raw, g_raw, o_raw = np.split(gates, 4, axis=1)

        f = sigmoid(f_raw)
        i = sigmoid(i_raw)
        g = np.tanh(g_raw)
        o = sigmoid(o_raw)

        c_next = f * c + i * g
        h_next = o * np.tanh(c_next)

        return h_next, c_next, {
            "forget": f,
            "input": i,
            "candidate": g,
            "output": o,
        }

    def forward(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence = np.asarray(x, dtype=float)
        if sequence.ndim != 3 or sequence.shape[2] != self.input_size:
            raise ValueError("expected (B,T,input_size)")

        batch, timesteps, _ = sequence.shape
        h = np.zeros((batch, self.hidden_size), dtype=float)
        c = np.zeros((batch, self.hidden_size), dtype=float)
        outputs = np.empty((batch, timesteps, self.hidden_size), dtype=float)

        for t in range(timesteps):
            h, c, _ = self.step(sequence[:, t, :], h, c)
            outputs[:, t, :] = h

        return outputs, h, c


@dataclass
class GRUCellNumPy:
    input_size: int
    hidden_size: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("sizes must be positive")

        rng = np.random.default_rng(self.seed)
        sx = 1.0 / np.sqrt(self.input_size)
        sh = 1.0 / np.sqrt(self.hidden_size)

        self.w_xr = rng.normal(0.0, sx, (self.input_size, self.hidden_size))
        self.w_hr = rng.normal(0.0, sh, (self.hidden_size, self.hidden_size))
        self.b_r = np.zeros(self.hidden_size)

        self.w_xz = rng.normal(0.0, sx, (self.input_size, self.hidden_size))
        self.w_hz = rng.normal(0.0, sh, (self.hidden_size, self.hidden_size))
        self.b_z = np.zeros(self.hidden_size)

        self.w_xn = rng.normal(0.0, sx, (self.input_size, self.hidden_size))
        self.w_hn = rng.normal(0.0, sh, (self.hidden_size, self.hidden_size))
        self.b_n = np.zeros(self.hidden_size)

    def step(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = np.asarray(x_t, dtype=float)
        h = np.asarray(h_prev, dtype=float)
        batch = x.shape[0]

        if x.shape != (batch, self.input_size):
            raise ValueError("x_t shape mismatch")
        if h.shape != (batch, self.hidden_size):
            raise ValueError("h_prev shape mismatch")

        r = sigmoid(x @ self.w_xr + h @ self.w_hr + self.b_r)
        z = sigmoid(x @ self.w_xz + h @ self.w_hz + self.b_z)
        n = np.tanh(x @ self.w_xn + (r * h) @ self.w_hn + self.b_n)
        h_next = (1.0 - z) * n + z * h

        return h_next, {"reset": r, "update": z, "candidate": n}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sequence = np.asarray(x, dtype=float)
        if sequence.ndim != 3 or sequence.shape[2] != self.input_size:
            raise ValueError("expected (B,T,input_size)")

        batch, timesteps, _ = sequence.shape
        h = np.zeros((batch, self.hidden_size), dtype=float)
        outputs = np.empty((batch, timesteps, self.hidden_size), dtype=float)

        for t in range(timesteps):
            h, _ = self.step(sequence[:, t, :], h)
            outputs[:, t, :] = h

        return outputs, h
