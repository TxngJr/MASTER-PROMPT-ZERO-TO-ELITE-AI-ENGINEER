from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "rag_numpy.py"
SPEC = importlib.util.spec_from_file_location("rag_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_chunk_words_overlap() -> None:
    chunks = mod.chunk_words(
        "a b c d e f g",
        chunk_size=4,
        overlap=2,
    )
    assert chunks == ["a b c d", "c d e f", "e f g"]


def test_cosine_top_k() -> None:
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    )
    ids, scores = mod.cosine_top_k(
        np.array([1.0, 0.0]),
        vectors,
        2,
    )
    assert ids.tolist() == [0, 2]
    assert scores[0] >= scores[1]


def test_rrf_promotes_consistent_item() -> None:
    fused = mod.reciprocal_rank_fusion(
        [
            ["a", "b", "c"],
            ["b", "a", "d"],
        ],
        rank_constant=10,
    )
    assert fused[0][0] in {"a", "b"}
    assert {item for item, _ in fused[:2]} == {"a", "b"}


def test_mmr_returns_unique_candidates() -> None:
    candidates = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ]
    )
    selected = mod.maximal_marginal_relevance(
        np.array([1.0, 0.0]),
        candidates,
        2,
        lambda_relevance=0.5,
    )
    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_context_packing_preserves_source_metadata() -> None:
    chunks = [
        {"source_id": "s1", "text": "one two three"},
        {"source_id": "s2", "text": "four five six seven"},
    ]
    packed = mod.pack_context(chunks, max_words=5)

    assert packed == [chunks[0]]
    assert packed[0]["source_id"] == "s1"


def test_retrieval_metrics() -> None:
    recall = mod.retrieval_recall_at_k(
        ["a", "x", "b"],
        {"a", "b"},
        2,
    )
    assert recall == 0.5

    mrr = mod.mean_reciprocal_rank(
        [["x", "a"], ["b", "x"]],
        [{"a"}, {"b"}],
    )
    np.testing.assert_allclose(mrr, 0.75)
