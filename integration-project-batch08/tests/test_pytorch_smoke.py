from pathlib import Path
import importlib.util

import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "cnn_framework_lab.py"
SPEC = importlib.util.spec_from_file_location("cnn_framework_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_pytorch_cnn_smoke_train() -> None:
    splits = lab.load_digits_splits(seed=3, limit=500)
    report = lab.run_pytorch(
        splits,
        epochs=1,
        batch_size=64,
        seed=3,
    )

    assert report["framework"] == "pytorch"
    assert report["parameter_count"] > 0
    assert 0.0 <= report["best_validation_accuracy"] <= 1.0
    assert 0.0 <= report["test_accuracy"] <= 1.0
