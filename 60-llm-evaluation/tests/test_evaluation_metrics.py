from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "evaluation_metrics.py"
SPEC = importlib.util.spec_from_file_location("evaluation_metrics_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_exact_match_normalizes_case_spacing_punctuation() -> None:
    score = mod.exact_match(
        [" Hello,   WORLD! "],
        ["hello world"],
    )
    assert score == 1.0


def test_token_f1_partial_overlap() -> None:
    score = mod.token_f1(
        ["red green blue"],
        ["green blue yellow"],
    )
    np.testing.assert_allclose(score, 2 / 3)


def test_brier_prefers_better_probabilities() -> None:
    outcomes = np.array([1, 0])
    good = mod.binary_brier_score(
        np.array([0.9, 0.1]),
        outcomes,
    )
    bad = mod.binary_brier_score(
        np.array([0.1, 0.9]),
        outcomes,
    )
    assert good < bad


def test_ece_perfect_binary_confidence() -> None:
    score = mod.expected_calibration_error(
        np.array([0.0, 1.0]),
        np.array([0, 1]),
        num_bins=2,
    )
    np.testing.assert_allclose(score, 0.0)


def test_bootstrap_is_deterministic_with_seed() -> None:
    values = np.array([0.0, 1.0, 1.0, 1.0])

    first = mod.bootstrap_mean_interval(
        values,
        num_resamples=200,
        seed=7,
    )
    second = mod.bootstrap_mean_interval(
        values,
        num_resamples=200,
        seed=7,
    )
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_paired_difference() -> None:
    a = np.array([1, 1, 1, 0], dtype=float)
    b = np.array([0, 1, 0, 0], dtype=float)

    mean, lower, upper = mod.paired_bootstrap_difference(
        a,
        b,
        num_resamples=200,
        seed=2,
    )
    assert mean > 0
    assert lower <= mean <= upper


def test_exact_contamination_rate() -> None:
    rate = mod.exact_contamination_rate(
        ["Example A", "new example"],
        ["example a!", "train only"],
    )
    assert rate == 0.5
