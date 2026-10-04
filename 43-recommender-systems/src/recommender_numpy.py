"""Educational recommender-system primitives implemented with NumPy."""

from __future__ import annotations

import numpy as np


def user_item_matrix(
    interactions: list[tuple[int, int, float]],
    *,
    num_users: int,
    num_items: int,
) -> np.ndarray:
    if num_users <= 0 or num_items <= 0:
        raise ValueError("dimensions must be positive")

    matrix = np.zeros((num_users, num_items), dtype=float)

    for user_id, item_id, value in interactions:
        if not 0 <= user_id < num_users:
            raise ValueError("user id out of range")
        if not 0 <= item_id < num_items:
            raise ValueError("item id out of range")
        matrix[user_id, item_id] += float(value)

    return matrix


def popularity_scores(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be 2D")
    return np.sum(values > 0, axis=0).astype(float)


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 2:
        raise ValueError("x must be a matrix")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def matrix_factorization_sgd(
    interactions: list[tuple[int, int, float]],
    *,
    num_users: int,
    num_items: int,
    factors: int = 8,
    epochs: int = 20,
    learning_rate: float = 0.03,
    regularization: float = 1e-3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if min(num_users, num_items, factors, epochs) <= 0:
        raise ValueError("dimensions/epochs must be positive")
    if learning_rate <= 0 or regularization < 0:
        raise ValueError("invalid optimization hyperparameters")
    if not interactions:
        raise ValueError("interactions must be non-empty")

    rng = np.random.default_rng(seed)
    users = rng.normal(0.0, 0.1, size=(num_users, factors))
    items = rng.normal(0.0, 0.1, size=(num_items, factors))

    losses: list[float] = []

    for _ in range(epochs):
        order = rng.permutation(len(interactions))
        squared_error = 0.0

        for index in order:
            user_id, item_id, rating = interactions[int(index)]

            if not 0 <= user_id < num_users:
                raise ValueError("user id out of range")
            if not 0 <= item_id < num_items:
                raise ValueError("item id out of range")

            user_vector = users[user_id].copy()
            item_vector = items[item_id].copy()

            prediction = float(user_vector @ item_vector)
            error = prediction - float(rating)
            squared_error += error**2

            users[user_id] -= learning_rate * (
                error * item_vector
                + regularization * user_vector
            )
            items[item_id] -= learning_rate * (
                error * user_vector
                + regularization * item_vector
            )

        losses.append(squared_error / len(interactions))

    return users, items, losses


def precision_at_k(
    ranked_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked_items[:k]
    if not top:
        return 0.0
    hits = sum(item in relevant_items for item in top)
    return float(hits / k)


def recall_at_k(
    ranked_items: list[int],
    relevant_items: set[int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_items:
        raise ValueError("relevant_items must be non-empty")
    hits = sum(item in relevant_items for item in ranked_items[:k])
    return float(hits / len(relevant_items))


def ndcg_at_k(
    ranked_items: list[int],
    relevance: dict[int, float],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        dcg += relevance.get(item, 0.0) / np.log2(rank + 1)

    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(
        value / np.log2(rank + 1)
        for rank, value in enumerate(ideal, start=1)
    )

    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def bpr_pair_loss(
    positive_score: float,
    negative_score: float,
) -> float:
    difference = float(positive_score - negative_score)
    return float(np.logaddexp(0.0, -difference))
