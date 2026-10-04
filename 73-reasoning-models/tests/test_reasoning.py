from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "reasoning.py"
SPEC = importlib.util.spec_from_file_location("reasoning_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_majority_vote_is_deterministic_on_tie() -> None:
    answer, count = mod.majority_vote(
        ["b", "a", "b", "a"]
    )
    assert answer == "a"
    assert count == 2


def test_self_consistency_confidence() -> None:
    confidence = mod.self_consistency_confidence(
        ["42", "42", "41", "42"]
    )
    np.testing.assert_allclose(confidence, 0.75)


def test_best_of_n() -> None:
    answer, score, index = mod.best_of_n(
        ["a", "b", "c"],
        [0.2, 0.9, 0.4],
    )
    assert answer == "b"
    assert score == 0.9
    assert index == 1


def test_pass_at_k() -> None:
    estimate = mod.pass_at_k(
        total_samples=10,
        correct_samples=2,
        k=3,
    )
    expected = 1.0 - (8 * 7 * 6) / (10 * 9 * 8)
    np.testing.assert_allclose(estimate, expected)


def test_allocate_sample_budget_exact_total() -> None:
    allocations = mod.allocate_sample_budget(
        total_token_budget=100,
        samples=6,
        reserve_tokens=10,
    )
    assert sum(allocations) == 90
    assert max(allocations) - min(allocations) <= 1


def test_search_efficiency() -> None:
    report = mod.search_efficiency(
        solved=4,
        attempted=5,
        generated_tokens=2000,
    )
    assert report["solve_rate"] == 0.8
    assert report["solved_per_1k_tokens"] == 2.0
    assert report["tokens_per_solved"] == 500.0


def test_weighted_vote() -> None:
    answer, score = mod.weighted_vote(
        ["x", "y", "x"],
        [0.2, 0.9, 0.8],
    )
    assert answer == "x"
    np.testing.assert_allclose(score, 1.0)
