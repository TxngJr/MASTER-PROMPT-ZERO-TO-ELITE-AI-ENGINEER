from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "activations_losses.py"
SPEC = importlib.util.spec_from_file_location("activations_losses_course", MODULE_PATH)
assert SPEC and SPEC.loader
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def finite_difference(fn, value, eps=1e-6):
    value = np.asarray(value, dtype=float)
    grad = np.zeros_like(value)

    for index in np.ndindex(value.shape):
        plus = value.copy()
        minus = value.copy()
        plus[index] += eps
        minus[index] -= eps
        grad[index] = (fn(plus) - fn(minus)) / (2.0 * eps)

    return grad


def test_sigmoid_and_softmax_stable_on_extreme_values() -> None:
    sigmoid = m.stable_sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(sigmoid))
    assert sigmoid[0] < 1e-10
    assert sigmoid[2] > 1.0 - 1e-10

    probabilities = m.softmax(
        np.array([[10000.0, 10001.0, 10002.0]]),
        axis=1,
    )
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_bce_with_logits_extreme_values_finite() -> None:
    logits = np.array([-1000.0, 1000.0])
    target = np.array([0.0, 1.0])

    loss, gradient = m.bce_with_logits_loss_and_grad(logits, target)

    assert np.isfinite(loss)
    assert np.all(np.isfinite(gradient))
    assert loss < 1e-10


def test_bce_gradient_matches_finite_difference() -> None:
    logits = np.array([-1.2, 0.3, 2.0])
    target = np.array([0.0, 1.0, 1.0])

    _, gradient = m.bce_with_logits_loss_and_grad(logits, target)
    numeric = finite_difference(
        lambda x: m.bce_with_logits_loss_and_grad(x, target)[0],
        logits,
    )

    np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-7)


def test_cross_entropy_gradient_matches_finite_difference() -> None:
    logits = np.array(
        [[1.2, -0.3, 0.8], [-0.5, 1.5, 0.1]],
        dtype=float,
    )
    target = np.array([2, 1])

    _, gradient = m.cross_entropy_with_logits_loss_and_grad(
        logits,
        target,
        label_smoothing=0.1,
    )
    numeric = finite_difference(
        lambda z: m.cross_entropy_with_logits_loss_and_grad(
            z,
            target,
            label_smoothing=0.1,
        )[0],
        logits,
    )

    np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-7)


def test_huber_is_less_quadratic_for_large_error() -> None:
    prediction = np.array([10.0])
    target = np.array([0.0])

    mse, _ = m.mse_loss_and_grad(prediction, target)
    huber, grad = m.huber_loss_and_grad(prediction, target, delta=1.0)

    assert huber < mse
    assert abs(grad[0]) == 1.0
