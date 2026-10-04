from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "data_pipeline.py"
SPEC = importlib.util.spec_from_file_location("data_pipeline_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_normalize_text_is_deterministic() -> None:
    text = "hello  \r\nworld   \n\n\nnext"
    first = mod.normalize_text(text)
    second = mod.normalize_text(text)

    assert first == second
    assert "\r" not in first
    assert "\n\n\n" not in first


def test_exact_deduplicate_after_normalization() -> None:
    documents = [
        {"document_id": "a", "text": "hello\r\nworld"},
        {"document_id": "b", "text": "hello\nworld"},
        {"document_id": "c", "text": "different"},
    ]

    kept, removed = mod.exact_deduplicate(documents)

    assert [doc["document_id"] for doc in kept] == ["a", "c"]
    assert removed == ["b"]


def test_deterministic_split_is_stable() -> None:
    first = mod.deterministic_split("document-42")
    second = mod.deterministic_split("document-42")
    assert first == second
    assert first in {"train", "validation", "test"}


def test_pack_and_shard_sequences() -> None:
    packed = mod.pack_token_documents(
        [[1, 2, 3], [4, 5, 6, 7]],
        sequence_length=4,
        separator_id=99,
    )
    assert packed.shape == (2, 4)
    np.testing.assert_array_equal(
        packed.reshape(-1),
        [1, 2, 3, 99, 4, 5, 6, 7],
    )

    shards = mod.shard_sequences(
        packed,
        sequences_per_shard=1,
    )
    assert len(shards) == 2
    assert all(shard.shape == (1, 4) for shard in shards)


def test_mixture_weights() -> None:
    weights = mod.normalize_mixture_weights(
        {"web": 8.0, "code": 2.0}
    )
    np.testing.assert_allclose(
        weights["web"] + weights["code"],
        1.0,
    )

    flattened = mod.temperature_mixture_weights(
        {"large": 0.9, "small": 0.1},
        alpha=0.5,
    )

    assert flattened["small"] > 0.1
    assert flattened["large"] < 0.9
