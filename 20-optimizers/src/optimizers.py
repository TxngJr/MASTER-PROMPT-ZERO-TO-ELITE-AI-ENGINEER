"""Small optimizer implementations operating on Tensor-like parameters."""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np


class Parameter(Protocol):
    data: np.ndarray
    grad: np.ndarray

    def zero_grad(self) -> None: ...


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer requires at least one parameter")
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for parameter in self.params:
            parameter.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-2,
        *,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.weight_decay = float(weight_decay)

    def step(self) -> None:
        for parameter in self.params:
            gradient = parameter.grad + self.weight_decay * parameter.data
            parameter.data -= self.lr * gradient


class Momentum(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-2,
        *,
        momentum: float = 0.9,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0,1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        self.momentum = float(momentum)
        self.nesterov = nesterov
        self.weight_decay = float(weight_decay)
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for index, parameter in enumerate(self.params):
            gradient = parameter.grad + self.weight_decay * parameter.data
            self.velocity[index] = (
                self.momentum * self.velocity[index] + gradient
            )
            if self.nesterov:
                update = gradient + self.momentum * self.velocity[index]
            else:
                update = self.velocity[index]
            parameter.data -= self.lr * update


class AdaGrad(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-2,
        *,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.sum_squares = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for index, parameter in enumerate(self.params):
            self.sum_squares[index] += parameter.grad**2
            parameter.data -= (
                self.lr
                * parameter.grad
                / (np.sqrt(self.sum_squares[index]) + self.eps)
            )


class RMSProp(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        *,
        beta: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0,1)")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.beta = float(beta)
        self.eps = float(eps)
        self.avg_squares = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for index, parameter in enumerate(self.params):
            self.avg_squares[index] = (
                self.beta * self.avg_squares[index]
                + (1.0 - self.beta) * parameter.grad**2
            )
            parameter.data -= (
                self.lr
                * parameter.grad
                / (np.sqrt(self.avg_squares[index]) + self.eps)
            )


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("betas must be in [0,1)")
        if eps <= 0 or weight_decay < 0:
            raise ValueError("invalid eps/weight_decay")

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def _gradient(self, parameter: Parameter) -> np.ndarray:
        return parameter.grad + self.weight_decay * parameter.data

    def step(self) -> None:
        self.t += 1

        for index, parameter in enumerate(self.params):
            gradient = self._gradient(parameter)
            self.m[index] = (
                self.beta1 * self.m[index]
                + (1.0 - self.beta1) * gradient
            )
            self.v[index] = (
                self.beta2 * self.v[index]
                + (1.0 - self.beta2) * gradient**2
            )

            m_hat = self.m[index] / (1.0 - self.beta1**self.t)
            v_hat = self.v[index] / (1.0 - self.beta2**self.t)

            parameter.data -= (
                self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            )


class AdamW(Adam):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(
            params,
            lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=0.0,
        )
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.decoupled_weight_decay = float(weight_decay)

    def step(self) -> None:
        if self.decoupled_weight_decay:
            factor = 1.0 - self.lr * self.decoupled_weight_decay
            for parameter in self.params:
                parameter.data *= factor

        super().step()
