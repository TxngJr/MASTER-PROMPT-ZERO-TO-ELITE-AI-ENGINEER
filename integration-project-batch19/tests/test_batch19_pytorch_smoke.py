from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "fine_tune_lora_sft_lab.py"
SPEC = importlib.util.spec_from_file_location("batch19_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_full_finetune_smoke() -> None:
    report = lab.run_full_finetune(
        steps=2,
        seed=5,
    )

    assert report["trainable_parameters"] > 0
    assert np.isfinite(report["final_adaptation_loss"])
    assert (
        report["final_adaptation_loss"]
        <= report["initial_adaptation_loss"]
    )


def test_lora_smoke() -> None:
    report = lab.run_lora(
        steps=2,
        rank=2,
        seed=5,
    )

    assert not report["base_weight_requires_grad"]
    assert 0.0 < report["trainable_fraction"] < 1.0
    assert np.isfinite(report["final_adaptation_loss"])
    assert (
        report["final_adaptation_loss"]
        <= report["initial_adaptation_loss"]
    )


def test_instruction_sft_smoke() -> None:
    report = lab.run_instruction_sft(
        steps=1,
        seed=5,
    )

    assert report["supervised_tokens"] > 0
    assert 0.0 < report["supervised_fraction"] < 1.0
    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["loss_last"])
