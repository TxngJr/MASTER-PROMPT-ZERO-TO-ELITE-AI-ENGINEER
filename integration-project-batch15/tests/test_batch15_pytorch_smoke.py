from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "recommendation_vision_retrieval_lab.py"
SPEC = importlib.util.spec_from_file_location("batch15_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_two_tower_smoke() -> None:
    report = lab.run_recommender(
        steps=2,
        seed=4,
    )

    assert report["parameter_count"] > 0
    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["loss_last"])
    assert 0.0 <= report["mean_recall_at_10"] <= 1.0


def test_segmentation_smoke() -> None:
    report = lab.run_segmentation(
        steps=2,
        seed=5,
    )

    assert report["parameter_count"] > 0
    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["loss_last"])
    assert 0.0 <= report["test_mean_iou"] <= 1.0
