"""Deterministic educational LLM data-pipeline primitives."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

import numpy as np


def normalize_text(
    text: str,
    *,
    unicode_form: str = "NFC",
) -> str:
    if unicode_form not in {"NFC", "NFKC"}:
        raise ValueError("unicode_form must be NFC or NFKC")

    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = unicodedata.normalize(unicode_form, value)

    lines = [line.rstrip() for line in value.split("\n")]
    value = "\n".join(lines).strip()
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value


def canonical_document_hash(
    text: str,
    *,
    unicode_form: str = "NFC",
) -> str:
    normalized = normalize_text(
        text,
        unicode_form=unicode_form,
    )
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def exact_deduplicate(
    documents: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    kept: list[dict[str, object]] = []
    removed_ids: list[str] = []
    seen_hashes: set[str] = set()

    for document in documents:
        if "document_id" not in document or "text" not in document:
            raise ValueError("documents need document_id and text")

        digest = canonical_document_hash(
            str(document["text"])
        )

        if digest in seen_hashes:
            removed_ids.append(str(document["document_id"]))
            continue

        seen_hashes.add(digest)
        kept.append(dict(document))

    return kept, removed_ids


def deterministic_split(
    document_id: str,
    *,
    train_fraction: float = 0.9,
    validation_fraction: float = 0.05,
) -> str:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("invalid train_fraction")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("invalid validation_fraction")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation must be < 1")

    digest = hashlib.sha256(
        document_id.encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64

    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + validation_fraction:
        return "validation"
    return "test"


def pack_token_documents(
    token_documents: Iterable[list[int]],
    *,
    sequence_length: int,
    separator_id: int | None = None,
    drop_remainder: bool = True,
) -> np.ndarray:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    stream: list[int] = []

    for document in token_documents:
        if stream and separator_id is not None:
            stream.append(int(separator_id))
        stream.extend(int(token) for token in document)

    if not stream:
        return np.empty(
            (0, sequence_length),
            dtype=np.int64,
        )

    if not drop_remainder:
        remainder = len(stream) % sequence_length
        if remainder:
            if separator_id is None:
                raise ValueError(
                    "separator_id required to pad when drop_remainder=False"
                )
            stream.extend(
                [int(separator_id)]
                * (sequence_length - remainder)
            )

    usable = len(stream) // sequence_length * sequence_length

    if usable == 0:
        return np.empty(
            (0, sequence_length),
            dtype=np.int64,
        )

    array = np.asarray(stream[:usable], dtype=np.int64)
    return array.reshape(-1, sequence_length)


def shard_sequences(
    sequences: np.ndarray,
    *,
    sequences_per_shard: int,
) -> list[np.ndarray]:
    values = np.asarray(sequences, dtype=np.int64)

    if values.ndim != 2:
        raise ValueError("sequences must be a matrix")
    if sequences_per_shard <= 0:
        raise ValueError("sequences_per_shard must be positive")

    return [
        values[start : start + sequences_per_shard].copy()
        for start in range(0, len(values), sequences_per_shard)
    ]


def normalize_mixture_weights(
    weights: dict[str, float],
) -> dict[str, float]:
    if not weights:
        raise ValueError("weights must be non-empty")
    if any(value < 0 for value in weights.values()):
        raise ValueError("weights must be non-negative")

    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("weights must have positive total")

    return {
        key: float(value / total)
        for key, value in weights.items()
    }


def temperature_mixture_weights(
    proportions: dict[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    base = normalize_mixture_weights(proportions)
    transformed = {
        key: value**alpha
        for key, value in base.items()
    }
    return normalize_mixture_weights(transformed)
