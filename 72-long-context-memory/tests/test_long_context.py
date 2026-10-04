from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "long_context.py"
SPEC = importlib.util.spec_from_file_location("long_context_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_dense_pairs() -> None:
    assert mod.causal_dense_attention_pairs(4) == 10


def test_sliding_window_reduces_long_attention_pairs() -> None:
    dense = mod.causal_dense_attention_pairs(100)
    sliding = mod.sliding_window_attention_pairs(
        100,
        window=10,
    )

    assert sliding < dense
    assert mod.attention_pair_reduction(
        100,
        window=10,
    ) > 1.0


def test_sliding_cache_is_bounded() -> None:
    assert mod.bounded_sliding_cache_tokens(
        100_000,
        window=4096,
    ) == 4096


def test_linear_rope_scaling() -> None:
    assert mod.linear_rope_scaled_position(
        8192,
        factor=4,
    ) == 2048.0


def test_chunk_ranges_overlap() -> None:
    ranges = mod.chunk_ranges(
        10,
        chunk_size=4,
        overlap=1,
    )
    assert ranges == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]


def test_context_selection_respects_budget() -> None:
    selected = mod.select_context_segments(
        [100, 50, 60],
        [0.9, 0.8, 0.7],
        token_budget=150,
    )
    assert selected == [0, 1]


def test_memory_score() -> None:
    score = mod.memory_score(
        similarity=1.0,
        recency=0.5,
        importance=0.0,
    )
    np.testing.assert_allclose(score, 0.7)


def test_retrieval_recall_at_k() -> None:
    recall = mod.retrieval_recall_at_k(
        ["a", "b", "c"],
        {"b", "c"},
        k=2,
    )
    assert recall == 0.5
