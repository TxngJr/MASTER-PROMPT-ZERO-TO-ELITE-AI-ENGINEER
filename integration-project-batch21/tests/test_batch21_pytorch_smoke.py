from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "quantized_inference_serving_lab.py"
SPEC = importlib.util.spec_from_file_location("batch21_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_torch_weight_only_smoke() -> None:
    report = lab.run_torch_weight_only(
        bits=4,
        seed=5,
    )

    assert report["outputs_finite"]
    assert np.isfinite(report["output_mse"])
    assert report["output_mse"] >= 0.0
    assert report["compression_vs_fp16"] == 4.0
