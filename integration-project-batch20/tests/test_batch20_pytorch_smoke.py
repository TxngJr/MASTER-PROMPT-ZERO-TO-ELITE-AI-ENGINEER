from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "rlhf_dpo_eval_lab.py"
SPEC = importlib.util.spec_from_file_location("batch20_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_reward_model_smoke() -> None:
    report = lab.run_reward_model(
        steps=10,
        seed=5,
    )

    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["loss_last"])
    assert report["loss_last"] < report["loss_first"]
    assert 0.0 <= report["pair_accuracy"] <= 1.0


def test_dpo_smoke() -> None:
    report = lab.run_dpo(
        steps=10,
        beta=0.5,
        seed=5,
    )

    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["loss_last"])
    assert report["loss_last"] < report["loss_first"]
    assert 0.0 <= report["preference_accuracy"] <= 1.0
    assert 0.0 <= report["true_best_accuracy"] <= 1.0
