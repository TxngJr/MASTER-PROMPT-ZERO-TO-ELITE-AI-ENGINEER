from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "tensor.py"
SPEC = importlib.util.spec_from_file_location("tiny_tensor", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)
Tensor = mod.Tensor


def numerical_gradient_scalar(fn, value, eps=1e-6):
    value = np.asarray(value, dtype=float)
    grad = np.zeros_like(value)

    for index in np.ndindex(value.shape):
        plus = value.copy()
        minus = value.copy()
        plus[index] += eps
        minus[index] -= eps
        grad[index] = (fn(plus) - fn(minus)) / (2.0 * eps)

    return grad


def test_scalar_chain_rule() -> None:
    x = Tensor(3.0, requires_grad=True)
    y = x * x + 2.0 * x
    y.backward()

    assert np.allclose(y.data, 15.0)
    assert np.allclose(x.grad, 8.0)


def test_branching_accumulates_gradient() -> None:
    x = Tensor(2.0, requires_grad=True)
    y = x * x + x
    y.backward()
    assert np.allclose(x.grad, 5.0)


def test_broadcast_bias_gradient() -> None:
    X = Tensor(np.ones((4, 3)), requires_grad=True)
    b = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

    loss = (X + b).sum()
    loss.backward()

    np.testing.assert_allclose(b.grad, [4.0, 4.0, 4.0])
    np.testing.assert_allclose(X.grad, np.ones((4, 3)))


def test_matmul_gradient_matches_numerical() -> None:
    rng = np.random.default_rng(4)
    X_data = rng.normal(size=(3, 2))
    W_data = rng.normal(size=(2, 4))

    X = Tensor(X_data, requires_grad=True)
    W = Tensor(W_data, requires_grad=True)
    loss = ((X @ W) ** 2).mean()
    loss.backward()

    numeric_W = numerical_gradient_scalar(
        lambda w: np.mean((X_data @ w) ** 2),
        W_data,
    )

    np.testing.assert_allclose(W.grad, numeric_W, rtol=1e-5, atol=1e-6)


def test_relu_sigmoid_tanh_gradients_are_finite() -> None:
    x = Tensor(np.array([[-1.0, 0.2, 1.5]]), requires_grad=True)
    out = (x.relu() + x.sigmoid() + x.tanh()).sum()
    out.backward()

    assert np.all(np.isfinite(x.grad))


def test_non_scalar_backward_requires_upstream_gradient() -> None:
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    y = x * 2.0

    try:
        y.backward()
    except ValueError:
        return
    raise AssertionError("non-scalar backward should require gradient")
