"""Educational NLP preprocessing and n-gram language-model utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
import unicodedata

import numpy as np


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def normalize_unicode(text: str, form: str = "NFC") -> str:
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError("unsupported Unicode normalization form")
    return unicodedata.normalize(form, text)


def simple_word_tokenize(
    text: str,
    *,
    lowercase: bool = False,
    normalization: str = "NFC",
) -> list[str]:
    normalized = normalize_unicode(text, normalization)
    if lowercase:
        normalized = normalized.casefold()
    return TOKEN_PATTERN.findall(normalized)


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    @classmethod
    def build(
        cls,
        token_sequences: list[list[str]],
        *,
        min_frequency: int = 1,
    ) -> "Vocabulary":
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")

        counts = Counter(
            token
            for sequence in token_sequences
            for token in sequence
        )

        mapping = {"<PAD>": 0, "<UNK>": 1}
        for token in sorted(counts):
            if counts[token] >= min_frequency and token not in mapping:
                mapping[token] = len(mapping)

        return cls(mapping)

    @property
    def id_to_token(self) -> dict[int, str]:
        return {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(token, unk) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        reverse = self.id_to_token
        return [reverse.get(idx, self.unk_token) for idx in ids]


def count_ngrams(
    tokens: list[str],
    n: int,
) -> Counter[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return Counter()

    return Counter(
        tuple(tokens[i : i + n])
        for i in range(len(tokens) - n + 1)
    )


@dataclass
class NGramLanguageModel:
    n: int = 2
    smoothing: float = 1.0

    def fit(self, tokens: list[str]) -> "NGramLanguageModel":
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if not tokens:
            raise ValueError("tokens must be non-empty")

        self.vocabulary_ = sorted(set(tokens))
        self.vocab_size_ = len(self.vocabulary_)
        self.ngram_counts_ = count_ngrams(tokens, self.n)

        self.context_counts_: dict[tuple[str, ...], int] = defaultdict(int)
        if self.n == 1:
            self.context_counts_[()] = len(tokens)
        else:
            for ngram, count in self.ngram_counts_.items():
                self.context_counts_[ngram[:-1]] += count
        return self

    def probability(
        self,
        token: str,
        context: tuple[str, ...] = (),
    ) -> float:
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError("fit before probability")

        if self.n == 1:
            context = ()
        elif len(context) != self.n - 1:
            raise ValueError("context length mismatch")

        key = context + (token,)
        numerator = self.ngram_counts_.get(key, 0) + self.smoothing
        denominator = (
            self.context_counts_.get(context, 0)
            + self.smoothing * self.vocab_size_
        )
        return float(numerator / denominator)

    def average_cross_entropy(self, tokens: list[str]) -> float:
        if len(tokens) < self.n:
            raise ValueError("token sequence too short")

        losses = []
        start = self.n - 1

        for i in range(start, len(tokens)):
            context = (
                tuple(tokens[i - self.n + 1 : i])
                if self.n > 1
                else ()
            )
            probability = self.probability(tokens[i], context)
            losses.append(-math.log(probability))

        return float(np.mean(losses))

    def perplexity(self, tokens: list[str]) -> float:
        return float(math.exp(self.average_cross_entropy(tokens)))


def make_next_token_windows(
    token_ids: list[int],
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if len(token_ids) <= context_length:
        raise ValueError("sequence must be longer than context_length")

    x = []
    y = []

    for start in range(len(token_ids) - context_length):
        window = token_ids[start : start + context_length + 1]
        x.append(window[:-1])
        y.append(window[1:])

    return np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)
