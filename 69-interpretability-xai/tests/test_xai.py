from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "xai.py"
SPEC = importlib.util.spec_from_file_location("xai_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_finite_difference_linear_gradient() -> None:
    weights = np.array([2.0, -3.0, 0.5])

    def function(x):
        return float(np.dot(weights, x))

    gradient = mod.finite_difference_gradient(
        function,
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(
        gradient,
        weights,
        atol=1e-8,
    )


def test_integrated_gradients_linear_completeness() -> None:
    weights = np.array([2.0, -3.0, 0.5])

    def function(x):
        return float(np.dot(weights, x) + 7.0)

    x = np.array([1.0, 2.0, 3.0])
    baseline = np.zeros(3)

    attrs = mod.integrated_gradients(
        function,
        x,
        baseline=baseline,
        steps=20,
    )

    expected = weights * x
    np.testing.assert_allclose(
        attrs,
        expected,
        atol=1e-7,
    )
    gap = mod.completeness_gap(
        function,
        x,
        attrs,
        baseline=baseline,
    )
    np.testing.assert_allclose(gap, 0.0, atol=1e-7)


def test_occlusion_linear() -> None:
    def function(x):
        return float(2 * x[0] + 3 * x[1])

    importance = mod.occlusion_importance(
        function,
        np.array([4.0, 5.0]),
    )
    np.testing.assert_allclose(
        importance,
        [8.0, 15.0],
    )


def test_exact_shapley_additive_game() -> None:
    contributions = np.array([1.0, 2.0, 4.0])

    def value(subset):
        return float(
            sum(contributions[index] for index in subset)
        )

    shapley = mod.exact_shapley_values(
        value,
        num_features=3,
    )
    np.testing.assert_allclose(
        shapley,
        contributions,
    )


def test_permutation_importance_finds_signal() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 2))
    y = 5.0 * x[:, 0]

    def predict(values):
        return 5.0 * values[:, 0]

    def negative_mse(targets, predictions):
        return -float(
            np.mean((targets - predictions) ** 2)
        )

    means, stds = mod.permutation_importance(
        predict,
        x,
        y,
        metric=negative_mse,
        repeats=5,
        seed=5,
    )
    assert means[0] > 1.0
    np.testing.assert_allclose(means[1], 0.0, atol=1e-12)
    assert np.all(stds >= 0)


def test_cosine_similarity() -> None:
    similarity = mod.cosine_attribution_similarity(
        np.array([1.0, 2.0]),
        np.array([2.0, 4.0]),
    )
    np.testing.assert_allclose(
        similarity,
        1.0,
        atol=1e-12,
    )
