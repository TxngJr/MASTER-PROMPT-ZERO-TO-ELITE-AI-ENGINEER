"""BERT-style data/objective helpers implemented with NumPy."""

from __future__ import annotations

import numpy as np


def add_special_tokens(
    first: list[int],
    *,
    cls_id: int,
    sep_id: int,
    second: list[int] | None = None,
) -> list[int]:
    output = [cls_id] + list(first) + [sep_id]
    if second is not None:
        output.extend(list(second))
        output.append(sep_id)
    return output


def create_token_type_ids(
    first_length: int,
    *,
    second_length: int | None = None,
) -> list[int]:
    if first_length < 0:
        raise ValueError("first_length must be non-negative")

    # [CLS] first [SEP]
    output = [0] * (first_length + 2)

    if second_length is not None:
        if second_length < 0:
            raise ValueError("second_length must be non-negative")
        # second [SEP]
        output.extend([1] * (second_length + 1))

    return output


def mlm_corrupt(
    token_ids: np.ndarray,
    *,
    vocab_size: int,
    mask_token_id: int,
    special_token_ids: set[int] | None = None,
    selection_probability: float = 0.15,
    mask_probability: float = 0.80,
    random_probability: float = 0.10,
    ignore_index: int = -100,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = np.asarray(token_ids, dtype=np.int64)

    if tokens.ndim != 1:
        raise ValueError("token_ids must be a 1D sequence")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not 0.0 <= selection_probability <= 1.0:
        raise ValueError("selection_probability must lie in [0,1]")
    if min(mask_probability, random_probability) < 0:
        raise ValueError("replacement probabilities must be non-negative")
    if mask_probability + random_probability > 1.0:
        raise ValueError("mask+random probabilities cannot exceed 1")

    special = special_token_ids or set()
    eligible = np.array(
        [token not in special for token in tokens],
        dtype=bool,
    )

    selected = (
        rng.random(tokens.shape[0]) < selection_probability
    ) & eligible

    corrupted = tokens.copy()
    labels = np.full(tokens.shape, ignore_index, dtype=np.int64)
    labels[selected] = tokens[selected]

    selected_indices = np.flatnonzero(selected)

    for index in selected_indices:
        draw = rng.random()

        if draw < mask_probability:
            corrupted[index] = mask_token_id
        elif draw < mask_probability + random_probability:
            corrupted[index] = int(rng.integers(0, vocab_size))
        else:
            # unchanged selected token
            pass

    return corrupted, labels, selected


def masked_accuracy(
    predicted_ids: np.ndarray,
    labels: np.ndarray,
    *,
    ignore_index: int = -100,
) -> float:
    predictions = np.asarray(predicted_ids)
    target = np.asarray(labels)

    if predictions.shape != target.shape:
        raise ValueError("predictions and labels must have equal shapes")

    mask = target != ignore_index
    if not np.any(mask):
        raise ValueError("no selected positions")

    return float(np.mean(predictions[mask] == target[mask]))
