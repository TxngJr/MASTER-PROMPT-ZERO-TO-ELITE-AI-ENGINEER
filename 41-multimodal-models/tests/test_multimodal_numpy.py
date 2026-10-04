from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "multimodal_numpy.py"
SPEC = importlib.util.spec_from_file_location("multimodal_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_l2_normalize() -> None:
    x = np.array([[3.0, 4.0], [1.0, 0.0]])
    normalized = mod.l2_normalize(x)
    np.testing.assert_allclose(
        np.linalg.norm(normalized, axis=1),
        1.0,
    )


def test_perfect_pairs_have_high_retrieval_accuracy() -> None:
    x = np.eye(4)
    assert mod.retrieval_top1_accuracy(x, x.copy()) == 1.0


def test_good_pairs_have_lower_contrastive_loss() -> None:
    good = np.eye(4)
    bad = good[::-1].copy()

    good_loss = mod.symmetric_contrastive_loss(
        good,
        good,
        temperature=0.1,
    )
    bad_loss = mod.symmetric_contrastive_loss(
        good,
        bad,
        temperature=0.1,
    )
    assert good_loss < bad_loss


def test_cross_attention_shapes() -> None:
    rng = np.random.default_rng(3)
    q = rng.normal(size=(2, 4, 6))
    k = rng.normal(size=(2, 5, 6))
    v = rng.normal(size=(2, 5, 7))

    output, weights = mod.simple_cross_attention(q, k, v)

    assert output.shape == (2, 4, 7)
    assert weights.shape == (2, 4, 5)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
