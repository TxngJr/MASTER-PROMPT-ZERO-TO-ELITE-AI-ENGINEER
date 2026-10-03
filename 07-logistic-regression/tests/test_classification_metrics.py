from pathlib import Path
import importlib.util

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


MODULE_PATH = Path(__file__).parents[1] / "src" / "classification_metrics.py"
SPEC = importlib.util.spec_from_file_location("classification_metrics", MODULE_PATH)
assert SPEC and SPEC.loader
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_binary_metrics_match_sklearn() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0])

    result = m.binary_metrics(y_true, y_pred)

    assert result["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert result["precision"] == pytest.approx(precision_score(y_true, y_pred))
    assert result["recall"] == pytest.approx(recall_score(y_true, y_pred))
    assert result["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert result["specificity"] == pytest.approx(2 / 3)
