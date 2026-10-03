from pathlib import Path
import importlib.util
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


MODULE_PATH = Path(__file__).parents[1] / "src" / "boosting.py"
SPEC = importlib.util.spec_from_file_location("boosting_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_gradient_boosting_reduces_training_loss() -> None:
    x = np.linspace(-2.0, 2.0, 120)
    X = x.reshape(-1, 1)
    y = x**2 + 0.2 * x

    model = mod.GradientBoostingRegressorFromScratch(
        n_estimators=60,
        learning_rate=0.1,
    ).fit(X, y)

    assert model.loss_history_[-1] < model.loss_history_[0]
    assert mean_squared_error(y, model.predict(X)) < 0.05


def test_gradient_boosting_reference_sklearn_is_useful() -> None:
    rng = np.random.default_rng(8)
    X = rng.uniform(-2.0, 2.0, size=(250, 2))
    y = X[:, 0] ** 2 - 0.5 * X[:, 1] + rng.normal(0, 0.1, 250)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=8,
    )

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=8,
    ).fit(X_train, y_train)

    assert mean_squared_error(y_test, model.predict(X_test)) < 0.2


def test_adaboost_from_scratch_learns_binary_problem() -> None:
    X, y = make_classification(
        n_samples=240,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        class_sep=1.4,
        random_state=13,
    )

    model = mod.AdaBoostBinaryClassifierFromScratch(
        n_estimators=30
    ).fit(X, y)

    accuracy = np.mean(model.predict(X) == y)
    assert accuracy > 0.85
    assert len(model.alphas_) >= 1


def test_sklearn_adaboost_reference_runs() -> None:
    X, y = make_classification(
        n_samples=220,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=6,
    )

    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=6),
        n_estimators=30,
        learning_rate=0.5,
        random_state=6,
    ).fit(X, y)

    assert model.score(X, y) > 0.85
