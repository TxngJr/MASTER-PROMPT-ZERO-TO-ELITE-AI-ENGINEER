"""Educational RAG retrieval, fusion, MMR and context-packing helpers."""

from __future__ import annotations

from collections import defaultdict
import numpy as np


def chunk_words(
    text: str,
    *,
    chunk_size: int,
    overlap: int = 0,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break

    return chunks


def _normalize(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def cosine_top_k(
    query: np.ndarray,
    vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=float).reshape(1, -1)
    x = np.asarray(vectors, dtype=float)

    if x.ndim != 2 or x.shape[1] != q.shape[1]:
        raise ValueError("dimension mismatch")
    if k <= 0:
        raise ValueError("k must be positive")

    scores = (_normalize(q) @ _normalize(x).T)[0]
    count = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")[:count]
    return order, scores[order]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    rank_constant: float = 60.0,
) -> list[tuple[str, float]]:
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (rank_constant + rank)

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )


def maximal_marginal_relevance(
    query: np.ndarray,
    candidates: np.ndarray,
    k: int,
    *,
    lambda_relevance: float = 0.7,
) -> list[int]:
    if not 0.0 <= lambda_relevance <= 1.0:
        raise ValueError("lambda_relevance must lie in [0,1]")
    if k <= 0:
        raise ValueError("k must be positive")

    x = _normalize(np.asarray(candidates, dtype=float))
    q = _normalize(np.asarray(query, dtype=float).reshape(1, -1))[0]

    if x.ndim != 2 or x.shape[1] != q.shape[0]:
        raise ValueError("dimension mismatch")

    relevance = x @ q
    selected: list[int] = []
    remaining = set(range(len(x)))

    while remaining and len(selected) < k:
        best_id = None
        best_score = -np.inf

        for candidate_id in sorted(remaining):
            if selected:
                redundancy = float(
                    np.max(x[candidate_id] @ x[selected].T)
                )
            else:
                redundancy = 0.0

            score = (
                lambda_relevance * relevance[candidate_id]
                - (1.0 - lambda_relevance) * redundancy
            )

            if score > best_score:
                best_score = score
                best_id = candidate_id

        assert best_id is not None
        selected.append(best_id)
        remaining.remove(best_id)

    return selected


def pack_context(
    chunks: list[dict[str, object]],
    *,
    max_words: int,
) -> list[dict[str, object]]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")

    packed: list[dict[str, object]] = []
    used = 0

    for chunk in chunks:
        if "text" not in chunk or "source_id" not in chunk:
            raise ValueError("each chunk needs text and source_id")

        text = str(chunk["text"])
        length = len(text.split())

        if used + length > max_words:
            continue

        packed.append(dict(chunk))
        used += length

    return packed


def retrieval_recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty")

    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return float(hits / len(relevant_ids))


def mean_reciprocal_rank(
    rankings: list[list[str]],
    relevant_sets: list[set[str]],
) -> float:
    if len(rankings) != len(relevant_sets) or not rankings:
        raise ValueError("rankings/relevant_sets mismatch or empty")

    reciprocals = []

    for ranking, relevant in zip(rankings, relevant_sets):
        reciprocal = 0.0

        for rank, item_id in enumerate(ranking, start=1):
            if item_id in relevant:
                reciprocal = 1.0 / rank
                break

        reciprocals.append(reciprocal)

    return float(np.mean(reciprocals))
