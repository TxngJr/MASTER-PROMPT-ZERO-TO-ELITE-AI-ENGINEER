from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "neural_foundations.py"
SPEC = importlib.util.spec_from_file_location("neural_foundations_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_activations_are_numerically_stable() -> None:
    values = np.array([-1000.0, 0.0, 1000.0])
    sig = mod.sigmoid(values)
    assert np.all(np.isfinite(sig))
    assert sig[0] < 1e-10
    assert sig[1] == 0.5
    assert sig[2] > 1.0 - 1e-10

    probs = mod.softmax(np.array([[1000.0, 1001.0, 1002.0]]), axis=1)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_linear_shape_and_parameter_count() -> None:
    layer = mod.Linear(3, 4, seed=1)
    X = np.ones((5, 3))
    output = layer(X)

    assert output.shape == (5, 4)
    assert layer.parameter_count == 3 * 4 + 4


def test_perceptron_learns_and_gate() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])

    model = mod.PerceptronClassifier(epochs=50).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)


def test_mlp_forward_shape_and_count() -> None:
    model = mod.TinyMLP(10, 8, 3, seed=7)
    X = np.ones((32, 10))
    output = model.forward(X)

    assert output.shape == (32, 3)
    assert model.parameter_count == (10 * 8 + 8) + (8 * 3 + 3)


def test_losses_prefer_better_predictions() -> None:
    y = np.array([0.0, 1.0])
    assert mod.binary_cross_entropy(y, np.array([0.1, 0.9])) < mod.binary_cross_entropy(
        y,
        np.array([0.9, 0.1]),
    )

    target = np.array([1.0, 2.0])
    assert mod.mse_loss(target, target) == 0.0
