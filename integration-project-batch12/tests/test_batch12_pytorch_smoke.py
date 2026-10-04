from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "bert_gpt_t5_lab.py"
SPEC = importlib.util.spec_from_file_location("batch12_transformer_families", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_tiny_bert_smoke() -> None:
    report = lab.run_bert(steps=1, seed=3)

    assert report["parameter_count"] > 0
    assert report["selected_positions"] > 0
    assert np.all(np.isfinite(report["loss_history"]))


def test_tiny_gpt_smoke() -> None:
    report = lab.run_gpt(steps=1, seed=4)

    assert report["parameter_count"] > 0
    assert np.all(np.isfinite(report["loss_history"]))
    assert len(report["generated_ids"]) == 8


def test_tiny_t5_smoke() -> None:
    report = lab.run_t5(steps=1, seed=5)

    assert report["parameter_count"] > 0
    assert np.all(np.isfinite(report["loss_history"]))
    assert len(report["prediction_ids"]) == 7
