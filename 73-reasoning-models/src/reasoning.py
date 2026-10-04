"""Reasoning-model evaluation, voting, verification and compute-budget utilities."""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Iterable


def majority_vote(answers: Iterable[str]) -> tuple[str, int]:
    values = list(answers)
    if not values:
        raise ValueError("answers must be non-empty")

    counts = Counter(values)
    best_count = max(counts.values())
    winners = sorted(
        answer
        for answer, count in counts.items()
        if count == best_count
    )
    return winners[0], int(best_count)


def self_consistency_confidence(
    answers: Iterable[str],
) -> float:
    values = list(answers)
    _, count = majority_vote(values)
    return float(count / len(values))


def best_of_n(
    candidates: list[str],
    verifier_scores: list[float],
) -> tuple[str, float, int]:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if len(candidates) != len(verifier_scores):
        raise ValueError("candidate/score length mismatch")

    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            verifier_scores[index],
            -index,
        ),
    )
    return (
        candidates[best_index],
        float(verifier_scores[best_index]),
        int(best_index),
    )


def pass_at_k(
    *,
    total_samples: int,
    correct_samples: int,
    k: int,
) -> float:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if not 0 <= correct_samples <= total_samples:
        raise ValueError("correct_samples out of range")
    if not 1 <= k <= total_samples:
        raise ValueError("k out of range")

    incorrect = total_samples - correct_samples
    if incorrect < k:
        return 1.0

    return float(
        1.0
        - comb(incorrect, k)
        / comb(total_samples, k)
    )


def allocate_sample_budget(
    *,
    total_token_budget: int,
    samples: int,
    reserve_tokens: int = 0,
) -> list[int]:
    if total_token_budget < 0:
        raise ValueError("total_token_budget must be non-negative")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens must be non-negative")
    if reserve_tokens > total_token_budget:
        raise ValueError("reserve exceeds total budget")

    usable = total_token_budget - reserve_tokens
    base = usable // samples
    remainder = usable % samples

    return [
        base + (1 if index < remainder else 0)
        for index in range(samples)
    ]


def verifier_accuracy(
    expected_correct: list[bool],
    predicted_correct: list[bool],
) -> float:
    if (
        len(expected_correct) != len(predicted_correct)
        or not expected_correct
    ):
        raise ValueError("matching non-empty labels required")

    matches = sum(
        expected == predicted
        for expected, predicted in zip(
            expected_correct,
            predicted_correct,
        )
    )
    return float(matches / len(expected_correct))


def search_efficiency(
    *,
    solved: int,
    attempted: int,
    generated_tokens: int,
) -> dict[str, float]:
    if attempted <= 0:
        raise ValueError("attempted must be positive")
    if not 0 <= solved <= attempted:
        raise ValueError("solved out of range")
    if generated_tokens <= 0:
        raise ValueError("generated_tokens must be positive")

    return {
        "solve_rate": float(solved / attempted),
        "solved_per_1k_tokens": float(
            solved / generated_tokens * 1000.0
        ),
        "tokens_per_solved": (
            float(generated_tokens / solved)
            if solved
            else float("inf")
        ),
    }


def weighted_vote(
    answers: list[str],
    scores: list[float],
) -> tuple[str, float]:
    if not answers:
        raise ValueError("answers must be non-empty")
    if len(answers) != len(scores):
        raise ValueError("answer/score length mismatch")
    if any(score < 0 for score in scores):
        raise ValueError("scores must be non-negative")

    totals: dict[str, float] = {}
    for answer, score in zip(answers, scores):
        totals[answer] = totals.get(answer, 0.0) + score

    best_score = max(totals.values())
    winners = sorted(
        answer
        for answer, score in totals.items()
        if score == best_score
    )
    return winners[0], float(best_score)
