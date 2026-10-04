from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "distributed_cuda_precision_lab.py"
SPEC = importlib.util.spec_from_file_location("batch18_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_gpu_inspection_cpu_or_cuda_safe() -> None:
    report = lab.run_gpu_inspection()

    assert "cuda_available" in report
    assert report["device_count"] >= 0

    if report["cuda_available"]:
        assert report["device_count"] >= 1
        assert report["devices"]


def test_mixed_precision_smoke() -> None:
    report = lab.run_mixed_precision(
        steps=1,
        seed=5,
    )

    assert report["parameter_count"] > 0
    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["loss_last"])
    assert report["output_finite"]
    assert report["autocast_dtype"] in {
        "torch.float16",
        "torch.bfloat16",
    }
