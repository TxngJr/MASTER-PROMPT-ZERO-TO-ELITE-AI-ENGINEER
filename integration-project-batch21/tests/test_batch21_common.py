from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "quantized_inference_serving_lab.py"
SPEC = importlib.util.spec_from_file_location("batch21_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_quantization_tradeoff() -> None:
    report = lab.run_quantization(seed=3)

    assert report["int8"]["ideal_weight_bytes"] > report["int4"]["ideal_weight_bytes"]
    assert report["int8"]["mse"] < report["int4"]["mse"]
    assert report["int4"]["compression_vs_fp16"] == 4.0


def test_capacity_gqa_reduces_cache() -> None:
    report = lab.run_capacity()

    assert report["gqa_kv_bytes"] < report["mha_kv_bytes"]
    assert report["gqa_reduction"] == 4.0
    assert report["paged_usage"]["blocks"] > 0
    assert 0.0 < report["padding_efficiency"] <= 1.0


def test_serving_report() -> None:
    report = lab.run_serving()

    assert report["p99_latency"] >= report["p50_latency"]
    assert report["required_replicas"] >= 1
    assert report["stable_requests"] + report["canary_requests"] == 10_000
    np.testing.assert_allclose(report["success_rate"], 0.995)
