from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")

MODULE_PATH = Path(__file__).parents[1] / "src" / "tiny_llm_capstone.py"
SPEC = importlib.util.spec_from_file_location("tiny_llm_capstone", MODULE_PATH)
assert SPEC and SPEC.loader
capstone = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capstone)


def test_local_dataset_has_train_and_validation_windows() -> None:
    built = capstone.build_local_dataset(sequence_length=8)
    assert len(built["x_train"]) > 0
    assert len(built["x_val"]) > 0
    probe = "ภาษาไทย + English"
    encoded = built["tokenizer"].encode(probe)
    assert built["tokenizer"].decode(encoded) == probe


def test_capstone_smoke_from_random_weights() -> None:
    report = capstone.run_capstone_smoke(
        pretrain_steps=1, sft_steps=1, preference_steps=1, seed=3
    )
    assert report["parameter_count"] > 0
    assert report["vocab_size"] > 256
    assert np.isfinite(report["pretrain_loss_first"])
    assert np.isfinite(report["validation_loss"])
    assert np.isfinite(report["preference_loss_last"])
    assert np.isfinite(report["quantization_mae"])
    assert report["checkpoint_tensor_keys"] > 0
