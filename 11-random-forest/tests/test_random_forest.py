from pathlib import Path
import importlib.util
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


MODULE_PATH = Path(__file__).parents[1] / "src" / "random_forest.py"
SPEC = importlib.util.spec_from_file_location("random_forest_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_bootstrap_forest_is_reproducible() -> None:
    X, y = make_classification(
        n_samples=180,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=4,
    )

    a = mod.RandomForestClassifierFromScratch(
        n_estimators=9,
        max_depth=4,
        seed=7,
    ).fit(X, y)
    b = mod.RandomForestClassifierFromScratch(
        n_estimators=9,
        max_depth=4,
        seed=7,
    ).fit(X, y)

    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert a.oob_score_ == b.oob_score_


def test_probability_rows_sum_to_one() -> None:
    X, y = make_classification(
        n_samples=160,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=2,
    )

    model = mod.RandomForestClassifierFromScratch(
        n_estimators=7,
        max_depth=3,
        seed=2,
    ).fit(X, y)

    probabilities = model.predict_proba(X[:20])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_from_scratch_forest_has_useful_test_accuracy() -> None:
    X, y = make_classification(
        n_samples=320,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        class_sep=1.3,
        random_state=12,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=12,
    )

    ours = mod.RandomForestClassifierFromScratch(
        n_estimators=15,
        max_depth=5,
        min_samples_leaf=2,
        seed=12,
    ).fit(X_train, y_train)

    accuracy = np.mean(ours.predict(X_test) == y_test)
    assert accuracy > 0.80
    assert ours.oob_score_ is not None
    assert 0.0 <= ours.oob_score_ <= 1.0


def test_sklearn_reference_model_runs_on_same_problem() -> None:
    X, y = make_classification(
        n_samples=250,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=21,
    )

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=5,
        random_state=21,
        oob_score=True,
        bootstrap=True,
        n_jobs=1,
    ).fit(X, y)

    assert model.score(X, y) > 0.9
    assert 0.0 <= model.oob_score_ <= 1.0
