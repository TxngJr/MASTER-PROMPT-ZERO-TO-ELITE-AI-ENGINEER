"""Instruction/chat formatting and assistant-only supervision utilities."""

from __future__ import annotations

import hashlib
from typing import Iterable


_ALLOWED_ROLES = {"system", "user", "assistant"}


def validate_messages(
    messages: list[dict[str, str]],
) -> None:
    if not messages:
        raise ValueError("messages must be non-empty")

    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise ValueError(
                f"message {index} must contain role/content only"
            )
        if message["role"] not in _ALLOWED_ROLES:
            raise ValueError(
                f"unsupported role: {message['role']}"
            )
        if not isinstance(message["content"], str):
            raise TypeError("message content must be a string")


def render_simple_chat(
    messages: list[dict[str, str]],
    *,
    end_of_turn: str = "<|eot|>",
) -> str:
    validate_messages(messages)

    parts = []
    for message in messages:
        parts.append(
            f"<|{message['role']}|>\n"
            f"{message['content']}{end_of_turn}"
        )
    return "\n".join(parts)


def concatenate_role_segments(
    segments: list[tuple[str, list[int]]],
) -> tuple[list[int], list[bool]]:
    token_ids: list[int] = []
    assistant_mask: list[bool] = []

    for role, tokens in segments:
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"unsupported role: {role}")

        token_ids.extend(int(token) for token in tokens)
        assistant_mask.extend(
            [role == "assistant"] * len(tokens)
        )

    return token_ids, assistant_mask


def assistant_only_labels(
    token_ids: list[int],
    assistant_mask: list[bool],
    *,
    ignore_index: int = -100,
) -> list[int]:
    if len(token_ids) != len(assistant_mask):
        raise ValueError("token/mask length mismatch")

    return [
        int(token) if supervised else int(ignore_index)
        for token, supervised in zip(token_ids, assistant_mask)
    ]


def supervised_token_count(
    labels: Iterable[int],
    *,
    ignore_index: int = -100,
) -> int:
    return sum(
        int(label) != ignore_index
        for label in labels
    )


def truncate_example(
    token_ids: list[int],
    labels: list[int],
    *,
    max_length: int,
    keep: str = "right",
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    if len(token_ids) != len(labels):
        raise ValueError("token/label length mismatch")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if keep not in {"left", "right"}:
        raise ValueError("keep must be left or right")

    if len(token_ids) <= max_length:
        result_ids = list(token_ids)
        result_labels = list(labels)
    elif keep == "right":
        result_ids = token_ids[-max_length:]
        result_labels = labels[-max_length:]
    else:
        result_ids = token_ids[:max_length]
        result_labels = labels[:max_length]

    if supervised_token_count(
        result_labels,
        ignore_index=ignore_index,
    ) == 0:
        raise ValueError(
            "truncation removed all supervised assistant tokens"
        )

    return result_ids, result_labels


def exact_deduplicate_examples(
    examples: list[list[dict[str, str]]],
) -> tuple[list[list[dict[str, str]]], list[int]]:
    kept = []
    removed_indices = []
    seen: set[str] = set()

    for index, messages in enumerate(examples):
        serialized = render_simple_chat(messages)
        digest = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

        if digest in seen:
            removed_indices.append(index)
            continue

        seen.add(digest)
        kept.append(messages)

    return kept, removed_indices
