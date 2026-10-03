from pathlib import Path
import importlib.util
import sys

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB


MODULE_PATH = Path(__file__).parents[1] / "src" / "naive_bayes.py"
SPEC = importlib.util.spec_from_file_location("naive_bayes_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_gaussian_nb_matches_sklearn_predictions() -> None:
    rng = np.random.default_rng(12)
    X0 = rng.normal(loc=[-2.0, 0.0], scale=[1.0, 0.7], size=(150, 2))
    X1 = rng.normal(loc=[2.0, 1.0], scale=[1.0, 0.7], size=(150, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))

    queries = rng.normal(size=(60, 2))

    ours = mod.GaussianNBFromScratch().fit(X, y)
    sk = GaussianNB().fit(X, y)

    agreement = np.mean(ours.predict(queries) == sk.predict(queries))
    assert agreement > 0.98


def test_multinomial_nb_matches_sklearn_predictions() -> None:
    X = np.array(
        [
            [5, 1, 0],
            [4, 2, 0],
            [6, 0, 1],
            [0, 1, 5],
            [1, 0, 4],
            [0, 2, 6],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    # Avoid exact posterior ties: ties are allowed to depend on deterministic
    # implementation details and are not useful for validating the formula.
    queries = np.array(
        [
            [3, 1, 0],
            [0, 1, 3],
            [2, 1, 1],
            [1, 1, 2],
        ],
        dtype=float,
    )

    ours = mod.MultinomialNBFromScratch(alpha=1.0).fit(X, y)
    sk = MultinomialNB(alpha=1.0).fit(X, y)

    np.testing.assert_array_equal(ours.predict(queries), sk.predict(queries))


def test_multinomial_rejects_negative_features() -> None:
    X = np.array([[1.0, -1.0], [2.0, 0.0]])
    y = np.array([0, 1])

    try:
        mod.MultinomialNBFromScratch().fit(X, y)
    except ValueError:
        return
    raise AssertionError("negative feature should be rejected")
