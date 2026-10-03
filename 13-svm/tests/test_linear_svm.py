from pathlib import Path
import importlib.util
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


MODULE_PATH = Path(__file__).parents[1] / "src" / "linear_svm.py"
SPEC = importlib.util.spec_from_file_location("linear_svm_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_hinge_loss_zero_for_large_correct_margin() -> None:
    y = np.array([-1, 1])
    scores = np.array([-2.0, 3.0])
    assert mod.hinge_loss(y, scores) == 0.0


def test_linear_svm_learns_separable_problem() -> None:
    rng = np.random.default_rng(5)
    X0 = rng.normal(loc=-2.0, scale=0.5, size=(100, 2))
    X1 = rng.normal(loc=2.0, scale=0.5, size=(100, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)

    Xs = StandardScaler().fit_transform(X)
    model = mod.LinearSVMFromScratch(
        C=2.0,
        learning_rate=0.03,
        steps=2500,
    ).fit(Xs, y)

    assert np.mean(model.predict(Xs) == y) > 0.98
    assert model.objective_history_[-1] < model.objective_history_[0]


def test_from_scratch_agrees_with_linear_svc_on_clear_data() -> None:
    X, y = make_classification(
        n_samples=350,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        class_sep=1.5,
        random_state=19,
    )
    Xs = StandardScaler().fit_transform(X)

    ours = mod.LinearSVMFromScratch(
        C=1.0,
        learning_rate=0.02,
        steps=3500,
    ).fit(Xs, y)

    ref = LinearSVC(C=1.0, max_iter=10000, random_state=19).fit(Xs, y)

    agreement = np.mean(ours.predict(Xs) == ref.predict(Xs))
    assert agreement > 0.90
