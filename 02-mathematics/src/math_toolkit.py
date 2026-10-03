"""Small from-scratch math helpers used before NumPy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import sqrt


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def _check_same_length(a: Vector, b: Vector) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} != {len(b)}")


def vector_add(a: Vector, b: Vector) -> list[float]:
    _check_same_length(a, b)
    return [x + y for x, y in zip(a, b)]


def dot(a: Vector, b: Vector) -> float:
    _check_same_length(a, b)
    return sum(x * y for x, y in zip(a, b))


def l2_norm(v: Vector) -> float:
    return sqrt(dot(v, v))


def transpose(matrix: Matrix) -> list[list[float]]:
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have equal length")
    return [list(column) for column in zip(*matrix)]


def matvec(matrix: Matrix, vector: Vector) -> list[float]:
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have equal length")
    if width != len(vector):
        raise ValueError(
            f"shape mismatch: matrix width {width}, vector length {len(vector)}"
        )
    return [dot(row, vector) for row in matrix]


def matmul(a: Matrix, b: Matrix) -> list[list[float]]:
    if not a or not b:
        return []
    a_width = len(a[0])
    b_width = len(b[0])
    if any(len(row) != a_width for row in a):
        raise ValueError("matrix a is ragged")
    if any(len(row) != b_width for row in b):
        raise ValueError("matrix b is ragged")
    if a_width != len(b):
        raise ValueError(
            f"shape mismatch: ({len(a)}x{a_width}) cannot multiply "
            f"({len(b)}x{b_width})"
        )
    b_t = transpose(b)
    return [[dot(row, column) for column in b_t] for row in a]


def mean(values: Vector) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def population_variance(values: Vector) -> float:
    if not values:
        raise ValueError("values must not be empty")
    center = mean(values)
    return sum((x - center) ** 2 for x in values) / len(values)


def sample_variance(values: Vector) -> float:
    if len(values) < 2:
        raise ValueError("sample variance requires at least two values")
    center = mean(values)
    return sum((x - center) ** 2 for x in values) / (len(values) - 1)


def covariance(x: Vector, y: Vector, *, sample: bool = True) -> float:
    _check_same_length(x, y)
    if not x:
        raise ValueError("values must not be empty")
    if sample and len(x) < 2:
        raise ValueError("sample covariance requires at least two pairs")

    x_mean = mean(x)
    y_mean = mean(y)
    denominator = len(x) - 1 if sample else len(x)
    return sum(
        (a - x_mean) * (b - y_mean) for a, b in zip(x, y)
    ) / denominator


def numerical_derivative(
    function: Callable[[float], float],
    x: float,
    *,
    h: float = 1e-6,
) -> float:
    if h <= 0:
        raise ValueError("h must be positive")
    return (function(x + h) - function(x - h)) / (2.0 * h)


def bayes(
    *,
    prior: float,
    likelihood: float,
    evidence: float,
) -> float:
    for name, value in {
        "prior": prior,
        "likelihood": likelihood,
        "evidence": evidence,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if evidence == 0:
        raise ValueError("evidence must be greater than zero")
    posterior = likelihood * prior / evidence
    if posterior > 1.0 + 1e-12:
        raise ValueError("inconsistent probabilities: posterior exceeds 1")
    return min(posterior, 1.0)
