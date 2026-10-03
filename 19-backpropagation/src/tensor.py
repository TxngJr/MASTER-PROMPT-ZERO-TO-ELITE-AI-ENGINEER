"""A tiny NumPy reverse-mode autodiff Tensor for education."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


ArrayLike = np.ndarray | float | int | list[float]


def _unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    grad = np.asarray(gradient, dtype=float)

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        *,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=float)
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @staticmethod
    def ensure(value: "Tensor | ArrayLike") -> "Tensor":
        return value if isinstance(value, Tensor) else Tensor(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=float)

    def __add__(self, other: "Tensor | ArrayLike") -> "Tensor":
        other_t = Tensor.ensure(other)
        out = Tensor(
            self.data + other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="+",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other_t.requires_grad:
                other_t.grad += _unbroadcast(out.grad, other_t.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: "Tensor | ArrayLike") -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: "Tensor | ArrayLike") -> "Tensor":
        return self + (-Tensor.ensure(other))

    def __rsub__(self, other: "Tensor | ArrayLike") -> "Tensor":
        return Tensor.ensure(other) - self

    def __mul__(self, other: "Tensor | ArrayLike") -> "Tensor":
        other_t = Tensor.ensure(other)
        out = Tensor(
            self.data * other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="*",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += _unbroadcast(
                    out.grad * other_t.data,
                    self.data.shape,
                )
            if other_t.requires_grad:
                other_t.grad += _unbroadcast(
                    out.grad * self.data,
                    other_t.data.shape,
                )

        out._backward = _backward
        return out

    def __rmul__(self, other: "Tensor | ArrayLike") -> "Tensor":
        return self * other

    def __truediv__(self, other: "Tensor | ArrayLike") -> "Tensor":
        return self * (Tensor.ensure(other) ** -1.0)

    def __rtruediv__(self, other: "Tensor | ArrayLike") -> "Tensor":
        return Tensor.ensure(other) / self

    def __pow__(self, exponent: float) -> "Tensor":
        if not isinstance(exponent, (int, float)):
            raise TypeError("power supports scalar exponents")
        out = Tensor(
            self.data**exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{exponent}",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * exponent * (self.data ** (exponent - 1.0))

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor | ArrayLike") -> "Tensor":
        other_t = Tensor.ensure(other)
        if self.data.ndim != 2 or other_t.data.ndim != 2:
            raise ValueError("educational matmul currently supports 2D arrays")

        out = Tensor(
            self.data @ other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="@",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad @ other_t.data.T
            if other_t.requires_grad:
                other_t.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if not self.requires_grad:
                return

            grad = out.grad
            if axis is None:
                expanded = np.broadcast_to(grad, self.data.shape)
            else:
                axes = (axis,) if isinstance(axis, int) else axis
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                if not keepdims:
                    for a in sorted(axes):
                        grad = np.expand_dims(grad, axis=a)
                expanded = np.broadcast_to(grad, self.data.shape)

            self.grad += expanded

        out._backward = _backward
        return out

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
            count = int(np.prod([self.data.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / float(count)

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        values = np.exp(self.data)
        out = Tensor(
            values,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * values

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        if np.any(self.data <= 0):
            raise ValueError("log requires positive values")
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        values = np.tanh(self.data)
        out = Tensor(
            values,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (1.0 - values**2)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        values = np.empty_like(self.data, dtype=float)
        positive = self.data >= 0
        values[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_values = np.exp(self.data[~positive])
        values[~positive] = exp_values / (1.0 + exp_values)

        out = Tensor(
            values,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * values * (1.0 - values)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        mask = self.data > 0.0
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * mask

        out._backward = _backward
        return out

    def backward(self, gradient: ArrayLike | None = None) -> None:
        if gradient is None:
            if self.data.size != 1:
                raise ValueError(
                    "non-scalar backward requires an explicit upstream gradient"
                )
            seed = np.ones_like(self.data, dtype=float)
        else:
            seed = np.asarray(gradient, dtype=float)
            if seed.shape != self.data.shape:
                raise ValueError("upstream gradient shape mismatch")

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            identity = id(node)
            if identity in visited:
                return
            visited.add(identity)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad += seed

        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data!r}, grad={self.grad!r}, "
            f"requires_grad={self.requires_grad})"
        )
