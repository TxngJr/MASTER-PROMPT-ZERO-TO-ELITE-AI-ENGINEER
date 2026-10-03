from pathlib import Path
import importlib.util

import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "math_toolkit.py"
SPEC = importlib.util.spec_from_file_location("math_toolkit", MODULE_PATH)
assert SPEC and SPEC.loader
math = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math)


def test_dot_and_norm() -> None:
    assert math.dot([1, 2, 3], [4, 5, 6]) == 32
    assert math.l2_norm([3, 4]) == 5


def test_matmul() -> None:
    result = math.matmul(
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    )
    assert result == [[19, 22], [43, 50]]


def test_shape_error() -> None:
    with pytest.raises(ValueError):
        math.matmul([[1, 2, 3]], [[1, 2], [3, 4]])


def test_statistics() -> None:
    values = [1.0, 2.0, 3.0]
    assert math.mean(values) == 2.0
    assert math.population_variance(values) == pytest.approx(2 / 3)
    assert math.sample_variance(values) == pytest.approx(1.0)


def test_covariance() -> None:
    assert math.covariance([1, 2, 3], [2, 4, 6]) == pytest.approx(2.0)


def test_numerical_derivative() -> None:
    derivative = math.numerical_derivative(lambda x: x * x, 3.0)
    assert derivative == pytest.approx(6.0, rel=1e-5)


def test_bayes() -> None:
    posterior = math.bayes(prior=0.01, likelihood=0.9, evidence=0.108)
    assert posterior == pytest.approx(1 / 12)
