from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier


MODULE_PATH = Path(__file__).parents[1] / "src" / "decision_tree.py"
SPEC = importlib.util.spec_from_file_location("decision_tree_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_impurity_values() -> None:
    pure = np.array([1, 1, 1, 1])
    mixed = np.array([0, 0, 1, 1])

    assert mod.gini_impurity(pure) == pytest.approx(0.0)
    assert mod.gini_impurity(mixed) == pytest.approx(0.5)
    assert mod.entropy(pure) == pytest.approx(0.0)
    assert mod.entropy(mixed) == pytest.approx(1.0)


def test_tree_learns_simple_boundary() -> None:
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = mod.DecisionTreeClassifierFromScratch(max_depth=2).fit(X, y)

    np.testing.assert_array_equal(tree.predict(X), y)


def test_tree_probability_rows_sum_to_one() -> None:
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = mod.DecisionTreeClassifierFromScratch(max_depth=1).fit(X, y)
    probabilities = tree.predict_proba(X)

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_matches_sklearn_on_clear_dataset() -> None:
    rng = np.random.default_rng(17)
    X = rng.normal(size=(250, 2))
    y = ((X[:, 0] > 0.2) & (X[:, 1] > -0.4)).astype(int)

    ours = mod.DecisionTreeClassifierFromScratch(max_depth=3).fit(X, y)
    sk = DecisionTreeClassifier(
        max_depth=3,
        criterion="gini",
        random_state=17,
    ).fit(X, y)

    agreement = np.mean(ours.predict(X) == sk.predict(X))
    assert agreement > 0.95
