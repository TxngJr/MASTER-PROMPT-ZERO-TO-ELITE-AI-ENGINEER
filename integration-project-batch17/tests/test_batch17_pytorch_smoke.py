from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "tokenizer_dataset_pretraining_lab.py"
SPEC = importlib.util.spec_from_file_location("batch17_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_tiny_pretraining_smoke() -> None:
    report = lab.run_pretraining(
        steps=1,
        seed=5,
    )

    assert report["parameter_count"] > 0
    assert report["vocab_size"] > 256
    assert report["training_sequences"] > 0
    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["perplexity_last"])
    assert report["tokens_seen"] > 0
    assert report["checkpoint_tensor_keys"] > 0
