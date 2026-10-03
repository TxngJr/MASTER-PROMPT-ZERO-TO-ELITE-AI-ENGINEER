from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "tiny_dl_lab.py"
SPEC = importlib.util.spec_from_file_location("tiny_dl_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lab
SPEC.loader.exec_module(lab)


def test_dataset_is_reproducible() -> None:
    X1, y1 = lab.make_dataset(300, seed=7)
    X2, y2 = lab.make_dataset(300, seed=7)

    np.testing.assert_allclose(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_tiny_framework_learns_nonlinear_problem() -> None:
    X, y = lab.make_dataset(420, seed=42)
    report = lab.run_experiment(
        X,
        y,
        seed=42,
        epochs=650,
    )

    assert (
        report["training"]["train_loss_final"]
        < report["training"]["train_loss_initial"]
    )
    assert report["validation"]["accuracy"] > 0.88
    assert report["test"]["accuracy"] > 0.88
    assert report["test"]["f1"] > 0.88


def test_parameter_count() -> None:
    model = lab.TinyMLPClassifier(hidden_features=16, seed=1)
    count = sum(parameter.data.size for parameter in model.parameters())
    assert count == (2 * 16 + 16) + (16 * 1 + 1)
