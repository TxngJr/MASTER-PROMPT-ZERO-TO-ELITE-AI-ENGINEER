"""Encoder-decoder and T5-style data helpers."""

from __future__ import annotations

import numpy as np


def shift_right(
    labels: list[int],
    *,
    start_token_id: int,
) -> list[int]:
    if not labels:
        raise ValueError("labels must be non-empty")
    return [start_token_id] + list(labels[:-1])


def make_padding_mask(
    token_ids: np.ndarray,
    *,
    pad_token_id: int,
) -> np.ndarray:
    tokens = np.asarray(token_ids)
    return tokens != pad_token_id


def span_corrupt(
    token_ids: list[int],
    spans: list[tuple[int, int]],
    *,
    sentinel_ids: list[int],
    eos_token_id: int,
) -> tuple[list[int], list[int]]:
    if len(spans) != len(sentinel_ids):
        raise ValueError("one sentinel id is required per span")

    length = len(token_ids)
    previous_end = 0

    for start, end in spans:
        if not (0 <= start < end <= length):
            raise ValueError("invalid span")
        if start < previous_end:
            raise ValueError("spans must be sorted and non-overlapping")
        previous_end = end

    corrupted: list[int] = []
    target: list[int] = []
    cursor = 0

    for (start, end), sentinel in zip(spans, sentinel_ids):
        corrupted.extend(token_ids[cursor:start])
        corrupted.append(sentinel)

        target.append(sentinel)
        target.extend(token_ids[start:end])

        cursor = end

    corrupted.extend(token_ids[cursor:])
    target.append(eos_token_id)

    return corrupted, target
