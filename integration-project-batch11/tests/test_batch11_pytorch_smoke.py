from pathlib import Path
import importlib.util

import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "vision_nlp_lab.py"
SPEC = importlib.util.spec_from_file_location("batch11_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_tiny_vit_smoke() -> None:
    report = lab.run_vit(
        epochs=1,
        batch_size=64,
        seed=5,
        limit=400,
    )

    assert report["parameter_count"] > 0
    assert report["patch_count"] == 16
    assert 0.0 <= report["best_validation_accuracy"] <= 1.0
    assert 0.0 <= report["test_accuracy"] <= 1.0
