from pathlib import Path
import importlib.util

import pytest

pytest.importorskip("tensorflow")


MODULE_PATH = Path(__file__).parents[1] / "src" / "cnn_framework_lab.py"
SPEC = importlib.util.spec_from_file_location("cnn_framework_lab_tf", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_tensorflow_cnn_smoke_train() -> None:
    splits = lab.load_digits_splits(seed=4, limit=500)
    report = lab.run_tensorflow(
        splits,
        epochs=1,
        batch_size=64,
        seed=4,
    )

    assert report["framework"] == "tensorflow"
    assert report["parameter_count"] > 0
    assert 0.0 <= report["best_validation_accuracy"] <= 1.0
    assert 0.0 <= report["test_accuracy"] <= 1.0
