from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "edge_ai.py"
SPEC = importlib.util.spec_from_file_location("edge_ai_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_tensor_storage_bytes() -> None:
    assert mod.tensor_storage_bytes(
        (10, 20),
        dtype="float32",
    ) == 800
    assert mod.tensor_storage_bytes(
        (10, 20),
        dtype="int8",
    ) == 200


def test_memory_budget() -> None:
    required = mod.peak_memory_bytes(
        model_bytes=1000,
        peak_activation_bytes=500,
        arena_bytes=200,
        runtime_overhead_bytes=100,
    )
    assert required == 1800
    assert mod.fits_memory_budget(
        required,
        available_bytes=2200,
        safety_fraction=0.9,
    )


def test_operator_coverage() -> None:
    report = mod.operator_coverage(
        ["Conv", "Relu", "Conv", "CustomOp"],
        {"Conv", "Relu"},
    )
    assert report["coverage"] == 2 / 3
    assert report["unsupported"] == ["CustomOp"]


def test_affine_int8_round_trip() -> None:
    x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    q, scale, zero = mod.affine_int8_quantize(x)
    reconstructed = mod.affine_int8_dequantize(
        q,
        scale=scale,
        zero_point=zero,
    )

    assert q.dtype == np.int8
    assert np.max(np.abs(reconstructed - x)) <= scale + 1e-12


def test_energy_metrics() -> None:
    energy = mod.energy_per_inference_mj(
        average_power_watts=2.0,
        latency_ms=10.0,
    )
    assert energy == 20.0
    assert mod.inferences_per_joule(
        energy_mj=energy,
    ) == 50.0


def test_sensor_window_and_duty_cycle() -> None:
    assert mod.sensor_window_samples(
        sample_rate_hz=16000,
        window_ms=20,
    ) == 320
    assert mod.duty_cycle(
        active_ms=2,
        period_ms=20,
    ) == 0.1
