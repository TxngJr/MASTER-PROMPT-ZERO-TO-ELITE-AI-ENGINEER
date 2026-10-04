from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "generative_transformer_lab.py"
SPEC = importlib.util.spec_from_file_location("batch10_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_gan_smoke() -> None:
    report = lab.run_gan(
        steps=2,
        batch_size=64,
        seed=3,
    )

    assert report["generator_parameters"] > 0
    assert report["discriminator_parameters"] > 0
    assert len(report["history"]) >= 1
    assert np.all(np.isfinite(report["generated_mean"]))


def test_transformer_smoke() -> None:
    report = lab.run_transformer(
        steps=2,
        batch_size=8,
        context_length=12,
        seed=4,
    )

    assert report["parameter_count"] > 0
    assert np.isfinite(report["best_validation_loss"])
    assert report["vocabulary_size"] > 1
    assert len(report["generated_text"]) > len("attention ")
