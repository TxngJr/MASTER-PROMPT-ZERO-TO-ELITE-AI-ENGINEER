"""Deterministic educational byte-level BPE tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Iterable


def bytes_to_base_tokens(data: bytes) -> list[int]:
    return list(data)


def pair_counts(tokens: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(tokens, tokens[1:]))


def merge_pair(
    tokens: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    output: list[int] = []
    index = 0

    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == pair[0]
            and tokens[index + 1] == pair[1]
        ):
            output.append(new_token_id)
            index += 2
        else:
            output.append(tokens[index])
            index += 1

    return output


def _token_bytes_from_merges(
    merges: list[tuple[int, int]],
) -> dict[int, bytes]:
    token_bytes = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    for rank, (left, right) in enumerate(merges):
        token_id = 256 + rank
        if left not in token_bytes or right not in token_bytes:
            raise ValueError("merge refers to unknown token")
        token_bytes[token_id] = (
            token_bytes[left] + token_bytes[right]
        )

    return token_bytes


def train_bpe(
    texts: Iterable[str],
    *,
    num_merges: int,
) -> list[tuple[int, int]]:
    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")

    sequences = [
        bytes_to_base_tokens(text.encode("utf-8"))
        for text in texts
    ]

    if not sequences:
        raise ValueError("training corpus must be non-empty")

    merges: list[tuple[int, int]] = []

    for merge_index in range(num_merges):
        counts: Counter[tuple[int, int]] = Counter()

        for sequence in sequences:
            counts.update(pair_counts(sequence))

        if not counts:
            break

        # Deterministic: highest count, then lexicographically smallest pair.
        best_pair = min(
            counts,
            key=lambda pair: (-counts[pair], pair),
        )
        new_token_id = 256 + merge_index

        sequences = [
            merge_pair(sequence, best_pair, new_token_id)
            for sequence in sequences
        ]
        merges.append(best_pair)

    return merges


@dataclass
class ByteBPETokenizer:
    merges: list[tuple[int, int]]

    def __post_init__(self) -> None:
        self.token_bytes = _token_bytes_from_merges(self.merges)
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode_bytes(self, data: bytes) -> list[int]:
        tokens = bytes_to_base_tokens(data)

        while len(tokens) >= 2:
            candidates = [
                (
                    self.merge_ranks[(tokens[index], tokens[index + 1])],
                    index,
                    (tokens[index], tokens[index + 1]),
                )
                for index in range(len(tokens) - 1)
                if (tokens[index], tokens[index + 1])
                in self.merge_ranks
            ]

            if not candidates:
                break

            best_rank = min(candidate[0] for candidate in candidates)
            pair = self.merges[best_rank]
            token_id = 256 + best_rank
            tokens = merge_pair(tokens, pair, token_id)

        return tokens

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode_bytes(self, token_ids: Iterable[int]) -> bytes:
        parts = []

        for token_id in token_ids:
            token = int(token_id)
            if token not in self.token_bytes:
                raise ValueError(f"unknown token id: {token}")
            parts.append(self.token_bytes[token])

        return b"".join(parts)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        errors: str = "strict",
    ) -> str:
        return self.decode_bytes(token_ids).decode(
            "utf-8",
            errors=errors,
        )

    def bytes_per_token(self, text: str) -> float:
        encoded = self.encode(text)
        if not encoded:
            return 0.0
        return len(text.encode("utf-8")) / len(encoded)
