from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "word_embeddings.py"
SPEC = importlib.util.spec_from_file_location("word_embeddings_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_skipgram_pairs_window_one() -> None:
    pairs = mod.skipgram_pairs([0, 1, 2], 1)
    assert pairs == [(0, 1), (1, 0), (1, 2), (2, 1)]


def test_negative_sampling_distribution_sums_to_one() -> None:
    distribution = mod.negative_sampling_distribution(
        [0, 0, 0, 1, 1, 2],
        vocab_size=3,
    )
    np.testing.assert_allclose(distribution.sum(), 1.0)
    assert distribution[0] > distribution[2]


def test_skipgram_updates_output_then_center_embeddings() -> None:
    model = mod.SkipGramNegativeSampling(
        vocab_size=5,
        embedding_dim=4,
        learning_rate=0.1,
        seed=3,
    )

    center_before = model.input_embeddings[1].copy()
    positive_before = model.output_embeddings[2].copy()

    first_loss = model.train_pair(
        center=1,
        positive_context=2,
        negative_contexts=np.array([3, 4]),
    )

    assert np.isfinite(first_loss)
    assert not np.allclose(
        positive_before,
        model.output_embeddings[2],
    )

    # Output vectors start at zero, so the first center-vector gradient is
    # exactly zero. After the first step, learned context vectors make the
    # next center update non-zero.
    np.testing.assert_allclose(
        center_before,
        model.input_embeddings[1],
    )

    second_loss = model.train_pair(
        center=1,
        positive_context=2,
        negative_contexts=np.array([3, 4]),
    )

    assert np.isfinite(second_loss)
    assert not np.allclose(
        center_before,
        model.input_embeddings[1],
    )


def test_cooccurrence_is_symmetric_for_symmetric_window() -> None:
    matrix = mod.cooccurrence_matrix(
        [0, 1, 2, 1],
        vocab_size=3,
        window_size=2,
    )
    np.testing.assert_allclose(matrix, matrix.T)


def test_glove_step_is_finite() -> None:
    model = mod.GloVeModel(
        vocab_size=4,
        embedding_dim=3,
        seed=4,
    )
    loss = model.train_entry(0, 1, count=5.0)
    assert np.isfinite(loss)


def test_fasttext_shared_subwords_make_related_vectors() -> None:
    table = mod.FastTextSubwordTable(
        embedding_dim=8,
        min_n=3,
        max_n=4,
        seed=7,
    ).fit_vocabulary(["play", "player", "playing"])

    play = table.vector("play")
    player = table.vector("player")
    unrelated = table.vector("zzzzz")

    assert np.linalg.norm(play) > 0
    assert np.linalg.norm(player) > 0
    assert mod.cosine_similarity(play, player) > mod.cosine_similarity(
        play,
        unrelated,
    )
