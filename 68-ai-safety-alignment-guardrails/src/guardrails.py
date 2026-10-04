"""Deterministic application-layer guardrail utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def risk_priority(
    *,
    likelihood: float,
    impact: float,
) -> float:
    if not 0.0 <= likelihood <= 1.0:
        raise ValueError("likelihood must lie in [0,1]")
    if not 0.0 <= impact <= 1.0:
        raise ValueError("impact must lie in [0,1]")
    return float(likelihood * impact)


def threshold_decision(
    score: float,
    *,
    allow_below: float,
    block_at_or_above: float,
) -> str:
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must lie in [0,1]")
    if not 0.0 <= allow_below <= block_at_or_above <= 1.0:
        raise ValueError("invalid thresholds")

    if score < allow_below:
        return "allow"
    if score >= block_at_or_above:
        return "block"
    return "review"


def budget_allows(
    *,
    used: int,
    requested: int,
    limit: int,
) -> bool:
    if min(used, requested, limit) < 0:
        raise ValueError("budget values must be non-negative")
    return used + requested <= limit


def permission_decision(
    *,
    tool: str,
    allowlisted_tools: Iterable[str],
    write_tools: Iterable[str] = (),
    user_approved: bool = False,
) -> str:
    allowed = set(allowlisted_tools)
    writes = set(write_tools)

    if tool not in allowed:
        return "deny"
    if tool in writes and not user_approved:
        return "approval_required"
    return "allow"


def retrieval_acl_filter(
    records: list[dict[str, Any]],
    *,
    principal_groups: set[str],
) -> list[dict[str, Any]]:
    authorized = []

    for record in records:
        groups = set(record.get("allowed_groups", []))
        if groups & principal_groups:
            authorized.append(record)

    return authorized


def validate_tool_arguments(
    arguments: dict[str, Any],
    *,
    required: dict[str, type],
    allowed_fields: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    allowed = allowed_fields or set(required)

    unknown = set(arguments) - allowed
    for field in sorted(unknown):
        errors.append(f"unknown field: {field}")

    for field, expected_type in required.items():
        if field not in arguments:
            errors.append(f"missing field: {field}")
            continue
        value = arguments[field]
        if not isinstance(value, expected_type):
            errors.append(
                f"wrong type for {field}: {type(value).__name__}"
            )

    return errors


def safety_confusion_counts(
    expected_block: list[bool],
    predicted_block: list[bool],
) -> dict[str, int]:
    if (
        len(expected_block) != len(predicted_block)
        or not expected_block
    ):
        raise ValueError("matching non-empty labels required")

    tp = fp = tn = fn = 0

    for expected, predicted in zip(
        expected_block,
        predicted_block,
    ):
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif not expected and not predicted:
            tn += 1
        else:
            fn += 1

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def safety_metrics(
    expected_block: list[bool],
    predicted_block: list[bool],
) -> dict[str, float]:
    counts = safety_confusion_counts(
        expected_block,
        predicted_block,
    )

    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    return {
        "block_recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "accuracy": float(accuracy),
    }
