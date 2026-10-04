"""Direct Preference Optimization math and diagnostics."""

from __future__ import annotations

import numpy as np


def preference_margin(
    chosen_log_probs: np.ndarray,
    rejected_log_probs: np.ndarray,
) -> np.ndarray:
    chosen = np.asarray(chosen_log_probs, dtype=float)
    rejected = np.asarray(rejected_log_probs, dtype=float)

    if chosen.shape != rejected.shape:
        raise ValueError("chosen/rejected shapes must match")

    return chosen - rejected


def dpo_logits(
    policy_chosen_log_probs: np.ndarray,
    policy_rejected_log_probs: np.ndarray,
    reference_chosen_log_probs: np.ndarray,
    reference_rejected_log_probs: np.ndarray,
    *,
    beta: float = 0.1,
) -> np.ndarray:
    if beta <= 0:
        raise ValueError("beta must be positive")

    policy_margin = preference_margin(
        policy_chosen_log_probs,
        policy_rejected_log_probs,
    )
    reference_margin = preference_margin(
        reference_chosen_log_probs,
        reference_rejected_log_probs,
    )

    if policy_margin.shape != reference_margin.shape:
        raise ValueError("policy/reference shapes must match")

    return beta * (policy_margin - reference_margin)


def dpo_loss(
    policy_chosen_log_probs: np.ndarray,
    policy_rejected_log_probs: np.ndarray,
    reference_chosen_log_probs: np.ndarray,
    reference_rejected_log_probs: np.ndarray,
    *,
    beta: float = 0.1,
) -> float:
    logits = dpo_logits(
        policy_chosen_log_probs,
        policy_rejected_log_probs,
        reference_chosen_log_probs,
        reference_rejected_log_probs,
        beta=beta,
    )

    if logits.size == 0:
        raise ValueError("preference batch must be non-empty")

    return float(np.mean(np.logaddexp(0.0, -logits)))


def dpo_pair_accuracy(
    policy_chosen_log_probs: np.ndarray,
    policy_rejected_log_probs: np.ndarray,
    reference_chosen_log_probs: np.ndarray,
    reference_rejected_log_probs: np.ndarray,
    *,
    beta: float = 0.1,
) -> float:
    logits = dpo_logits(
        policy_chosen_log_probs,
        policy_rejected_log_probs,
        reference_chosen_log_probs,
        reference_rejected_log_probs,
        beta=beta,
    )

    if logits.size == 0:
        raise ValueError("preference batch must be non-empty")

    return float(np.mean(logits > 0))


def chosen_rejected_rewards(
    policy_chosen_log_probs: np.ndarray,
    policy_rejected_log_probs: np.ndarray,
    reference_chosen_log_probs: np.ndarray,
    reference_rejected_log_probs: np.ndarray,
    *,
    beta: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    if beta <= 0:
        raise ValueError("beta must be positive")

    pc = np.asarray(policy_chosen_log_probs, dtype=float)
    pr = np.asarray(policy_rejected_log_probs, dtype=float)
    rc = np.asarray(reference_chosen_log_probs, dtype=float)
    rr = np.asarray(reference_rejected_log_probs, dtype=float)

    if not (pc.shape == pr.shape == rc.shape == rr.shape):
        raise ValueError("all log-probability shapes must match")

    return beta * (pc - rc), beta * (pr - rr)


def completion_logprob(
    token_log_probs: np.ndarray,
    completion_mask: np.ndarray,
) -> np.ndarray:
    log_probs = np.asarray(token_log_probs, dtype=float)
    mask = np.asarray(completion_mask, dtype=bool)

    if log_probs.shape != mask.shape:
        raise ValueError("log-probability/mask shapes must match")
    if log_probs.ndim < 1:
        raise ValueError("expected token dimension")

    return np.sum(
        np.where(mask, log_probs, 0.0),
        axis=-1,
    )


def length_audit(
    chosen_lengths: np.ndarray,
    rejected_lengths: np.ndarray,
) -> dict[str, float]:
    chosen = np.asarray(chosen_lengths, dtype=float)
    rejected = np.asarray(rejected_lengths, dtype=float)

    if chosen.shape != rejected.shape or chosen.size == 0:
        raise ValueError("matching non-empty lengths required")

    return {
        "chosen_mean": float(np.mean(chosen)),
        "rejected_mean": float(np.mean(rejected)),
        "mean_difference": float(np.mean(chosen - rejected)),
        "chosen_longer_fraction": float(np.mean(chosen > rejected)),
    }


def ipo_squared_loss(
    policy_chosen_log_probs: np.ndarray,
    policy_rejected_log_probs: np.ndarray,
    reference_chosen_log_probs: np.ndarray,
    reference_rejected_log_probs: np.ndarray,
    *,
    beta: float,
) -> float:
    """Educational IPO-style squared margin objective.

    Uses target gap 1/(2*beta), matching a common IPO presentation.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")

    policy_margin = preference_margin(
        policy_chosen_log_probs,
        policy_rejected_log_probs,
    )
    reference_margin = preference_margin(
        reference_chosen_log_probs,
        reference_rejected_log_probs,
    )

    gap = policy_margin - reference_margin
    target = 1.0 / (2.0 * beta)

    return float(np.mean((gap - target) ** 2))
