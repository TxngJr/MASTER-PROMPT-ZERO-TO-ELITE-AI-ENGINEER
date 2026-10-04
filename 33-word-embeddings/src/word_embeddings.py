"""Educational Word2Vec, GloVe and FastText-style embedding primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def skipgram_pairs(
    token_ids: list[int],
    window_size: int,
) -> list[tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    pairs: list[tuple[int, int]] = []

    for center_index, center in enumerate(token_ids):
        left = max(0, center_index - window_size)
        right = min(len(token_ids), center_index + window_size + 1)

        for context_index in range(left, right):
            if context_index == center_index:
                continue
            pairs.append((center, token_ids[context_index]))

    return pairs


def negative_sampling_distribution(
    token_ids: list[int],
    vocab_size: int,
    *,
    exponent: float = 0.75,
) -> np.ndarray:
    if vocab_size <= 0 or exponent <= 0:
        raise ValueError("invalid vocab_size/exponent")

    counts = np.bincount(
        np.asarray(token_ids, dtype=np.int64),
        minlength=vocab_size,
    ).astype(float)

    weights = counts**exponent
    total = weights.sum()

    if total <= 0:
        raise ValueError("cannot build distribution from empty counts")

    return weights / total


@dataclass
class SkipGramNegativeSampling:
    vocab_size: int
    embedding_dim: int
    learning_rate: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        if self.vocab_size <= 1 or self.embedding_dim <= 0:
            raise ValueError("invalid dimensions")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        rng = np.random.default_rng(self.seed)
        scale = 0.5 / self.embedding_dim
        self.input_embeddings = rng.uniform(
            -scale,
            scale,
            size=(self.vocab_size, self.embedding_dim),
        )
        self.output_embeddings = np.zeros(
            (self.vocab_size, self.embedding_dim),
            dtype=float,
        )

    def train_pair(
        self,
        center: int,
        positive_context: int,
        negative_contexts: np.ndarray,
    ) -> float:
        negatives = np.asarray(negative_contexts, dtype=np.int64).reshape(-1)

        if not 0 <= center < self.vocab_size:
            raise ValueError("center id out of range")
        if not 0 <= positive_context < self.vocab_size:
            raise ValueError("positive context id out of range")
        if np.any((negatives < 0) | (negatives >= self.vocab_size)):
            raise ValueError("negative context id out of range")

        center_vec = self.input_embeddings[center].copy()
        pos_vec = self.output_embeddings[positive_context].copy()
        neg_vecs = self.output_embeddings[negatives].copy()

        pos_score = float(center_vec @ pos_vec)
        neg_scores = neg_vecs @ center_vec

        pos_prob = float(sigmoid(pos_score))
        neg_prob = sigmoid(neg_scores)

        eps = 1e-12
        loss = -np.log(pos_prob + eps) - np.sum(
            np.log(1.0 - neg_prob + eps)
        )

        grad_center = (pos_prob - 1.0) * pos_vec
        if len(negatives):
            grad_center += np.sum(
                neg_prob[:, None] * neg_vecs,
                axis=0,
            )

        grad_pos = (pos_prob - 1.0) * center_vec
        grad_neg = neg_prob[:, None] * center_vec[None, :]

        self.input_embeddings[center] -= self.learning_rate * grad_center
        self.output_embeddings[positive_context] -= (
            self.learning_rate * grad_pos
        )

        for row, token_id in enumerate(negatives):
            self.output_embeddings[token_id] -= (
                self.learning_rate * grad_neg[row]
            )

        return float(loss)


def cooccurrence_matrix(
    token_ids: list[int],
    vocab_size: int,
    window_size: int,
    *,
    distance_weighting: bool = True,
) -> np.ndarray:
    if vocab_size <= 0 or window_size <= 0:
        raise ValueError("invalid dimensions")

    matrix = np.zeros((vocab_size, vocab_size), dtype=float)

    for i, center in enumerate(token_ids):
        left = max(0, i - window_size)
        right = min(len(token_ids), i + window_size + 1)

        for j in range(left, right):
            if i == j:
                continue
            context = token_ids[j]
            distance = abs(i - j)
            weight = 1.0 / distance if distance_weighting else 1.0
            matrix[center, context] += weight

    return matrix


@dataclass
class GloVeModel:
    vocab_size: int
    embedding_dim: int
    learning_rate: float = 0.03
    x_max: float = 100.0
    alpha: float = 0.75
    seed: int = 42

    def __post_init__(self) -> None:
        if self.vocab_size <= 1 or self.embedding_dim <= 0:
            raise ValueError("invalid dimensions")
        if min(self.learning_rate, self.x_max, self.alpha) <= 0:
            raise ValueError("hyperparameters must be positive")

        rng = np.random.default_rng(self.seed)
        self.word = rng.normal(
            0.0,
            0.1,
            (self.vocab_size, self.embedding_dim),
        )
        self.context = rng.normal(
            0.0,
            0.1,
            (self.vocab_size, self.embedding_dim),
        )
        self.word_bias = np.zeros(self.vocab_size)
        self.context_bias = np.zeros(self.vocab_size)

    def _weight(self, count: float) -> float:
        if count <= 0:
            return 0.0
        if count >= self.x_max:
            return 1.0
        return float((count / self.x_max) ** self.alpha)

    def train_entry(
        self,
        word_id: int,
        context_id: int,
        count: float,
    ) -> float:
        if count <= 0:
            raise ValueError("count must be positive")

        w = self.word[word_id].copy()
        c = self.context[context_id].copy()
        weight = self._weight(count)

        error = (
            float(w @ c)
            + self.word_bias[word_id]
            + self.context_bias[context_id]
            - np.log(count)
        )
        loss = weight * error**2
        gradient_scale = 2.0 * weight * error

        self.word[word_id] -= (
            self.learning_rate * gradient_scale * c
        )
        self.context[context_id] -= (
            self.learning_rate * gradient_scale * w
        )
        self.word_bias[word_id] -= (
            self.learning_rate * gradient_scale
        )
        self.context_bias[context_id] -= (
            self.learning_rate * gradient_scale
        )

        return float(loss)

    def combined_embeddings(self) -> np.ndarray:
        return self.word + self.context


def character_ngrams(
    word: str,
    *,
    min_n: int = 3,
    max_n: int = 6,
) -> list[str]:
    if min_n <= 0 or max_n < min_n:
        raise ValueError("invalid n-gram range")

    bounded = "<" + word + ">"
    grams: list[str] = []

    for n in range(min_n, max_n + 1):
        for start in range(len(bounded) - n + 1):
            grams.append(bounded[start : start + n])

    return grams


@dataclass
class FastTextSubwordTable:
    embedding_dim: int
    min_n: int = 3
    max_n: int = 6
    seed: int = 42

    def fit_vocabulary(self, words: list[str]) -> "FastTextSubwordTable":
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        grams = sorted({
            gram
            for word in words
            for gram in character_ngrams(
                word,
                min_n=self.min_n,
                max_n=self.max_n,
            )
        })

        rng = np.random.default_rng(self.seed)
        self.ngram_to_id_ = {
            gram: index for index, gram in enumerate(grams)
        }
        self.ngram_embeddings_ = rng.normal(
            0.0,
            1.0 / np.sqrt(self.embedding_dim),
            size=(len(grams), self.embedding_dim),
        )
        return self

    def vector(self, word: str) -> np.ndarray:
        if not hasattr(self, "ngram_to_id_"):
            raise RuntimeError("fit_vocabulary before vector")

        ids = [
            self.ngram_to_id_[gram]
            for gram in character_ngrams(
                word,
                min_n=self.min_n,
                max_n=self.max_n,
            )
            if gram in self.ngram_to_id_
        ]

        if not ids:
            return np.zeros(self.embedding_dim, dtype=float)

        return self.ngram_embeddings_[ids].mean(axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)

    if x.shape != y.shape:
        raise ValueError("vectors must have equal shapes")

    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return 0.0
    return float((x @ y) / denominator)
