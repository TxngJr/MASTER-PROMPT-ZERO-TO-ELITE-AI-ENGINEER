from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "diffusion_multimodal_speech_lab.py"
SPEC = importlib.util.spec_from_file_location("batch14_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_diffusion_smoke() -> None:
    report = lab.run_diffusion(
        steps=2,
        seed=3,
    )

    assert report["parameter_count"] > 0
    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["loss_last"])
    assert len(report["generated_mean"]) == 2


def test_multimodal_smoke() -> None:
    report = lab.run_multimodal(
        steps=2,
        seed=4,
    )

    assert report["parameter_count"] > 0
    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["loss_last"])
    assert 0.0 <= report["image_to_text_top1"] <= 1.0
    assert 0.0 <= report["text_to_image_top1"] <= 1.0


def test_speech_ctc_smoke() -> None:
    report = lab.run_speech_ctc(
        steps=1,
        seed=5,
    )

    assert report["parameter_count"] > 0
    assert np.isfinite(report["loss_first"])
    assert report["feature_shape"][0] == 48
    assert len(report["example_target"]) == 2
