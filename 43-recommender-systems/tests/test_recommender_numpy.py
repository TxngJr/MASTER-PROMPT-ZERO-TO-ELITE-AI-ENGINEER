from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "recommender_numpy.py"
SPEC = importlib.util.spec_from_file_location("recommender_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_user_item_matrix_accumulates() -> None:
    matrix = mod.user_item_matrix(
        [(0, 1, 1.0), (0, 1, 2.0), (1, 0, 3.0)],
        num_users=2,
        num_items=3,
    )
    assert matrix[0, 1] == 3.0
    assert matrix[1, 0] == 3.0


def test_cosine_similarity_diagonal() -> None:
    x = np.array([[1.0, 0.0], [1.0, 1.0]])
    similarities = mod.cosine_similarity_matrix(x)
    np.testing.assert_allclose(np.diag(similarities), 1.0)


def test_matrix_factorization_loss_finite() -> None:
    interactions = [
        (0, 0, 5.0),
        (0, 1, 4.0),
        (1, 0, 4.0),
        (1, 2, 5.0),
    ]
    users, items, losses = mod.matrix_factorization_sgd(
        interactions,
        num_users=2,
        num_items=3,
        factors=3,
        epochs=3,
        seed=4,
    )
    assert users.shape == (2, 3)
    assert items.shape == (3, 3)
    assert np.all(np.isfinite(losses))


def test_ranking_metrics() -> None:
    ranking = [4, 1, 3, 2]
    relevant = {1, 2}

    assert mod.precision_at_k(ranking, relevant, 2) == 0.5
    assert mod.recall_at_k(ranking, relevant, 2) == 0.5

    ndcg = mod.ndcg_at_k(
        ranking,
        {1: 2.0, 2: 1.0},
        4,
    )
    assert 0.0 <= ndcg <= 1.0


def test_bpr_loss_prefers_positive_above_negative() -> None:
    good = mod.bpr_pair_loss(3.0, 0.0)
    bad = mod.bpr_pair_loss(0.0, 3.0)
    assert good < bad
