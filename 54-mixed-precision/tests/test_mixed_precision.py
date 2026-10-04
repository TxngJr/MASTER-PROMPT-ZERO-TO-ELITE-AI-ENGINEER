from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "mixed_precision.py"
SPEC = importlib.util.spec_from_file_location("mixed_precision_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_bf16_has_fp32_like_exponent_range() -> None:
    fp32 = mod.floating_format_info("fp32")
    fp16 = mod.floating_format_info("fp16")
    bf16 = mod.floating_format_info("bf16")

    assert bf16["exponent_bits"] == fp32["exponent_bits"]
    assert bf16["exponent_bits"] > fp16["exponent_bits"]
    assert bf16["fraction_bits"] < fp16["fraction_bits"]


def test_bfloat16_roundtrip_reduces_precision() -> None:
    x = np.array([1.234567, -9.876543], dtype=np.float32)
    rounded = mod.bfloat16_roundtrip(x)

    assert rounded.dtype == np.float32
    assert np.all(np.isfinite(rounded))
    assert not np.array_equal(rounded, x)
    assert mod.relative_error(x, rounded) < 0.02


def test_scale_then_unscale_recovers_gradients() -> None:
    gradients = [
        np.array([1e-8, -2e-8]),
        np.array([3e-5]),
    ]
    scaled = mod.scale_gradients(gradients, 1024.0)
    restored = mod.unscale_gradients(scaled, 1024.0)

    for original, recovered in zip(gradients, restored):
        np.testing.assert_allclose(original, recovered)


def test_dynamic_loss_scale_backoff_and_growth() -> None:
    scale, stable = mod.update_dynamic_loss_scale(
        1024.0,
        found_nonfinite=True,
        stable_steps=10,
        growth_interval=2,
    )
    assert scale == 512.0
    assert stable == 0

    scale, stable = mod.update_dynamic_loss_scale(
        scale,
        found_nonfinite=False,
        stable_steps=1,
        growth_interval=2,
    )
    assert scale == 1024.0
    assert stable == 0


def test_storage_bytes() -> None:
    assert mod.tensor_storage_bytes(
        1000,
        dtype="fp32",
    ) == 4000
    assert mod.tensor_storage_bytes(
        1000,
        dtype="fp16",
    ) == 2000
    assert mod.tensor_storage_bytes(
        1000,
        dtype="bf16",
    ) == 2000


def test_nonfinite_detection() -> None:
    assert not mod.contains_nonfinite(
        [np.array([1.0, 2.0])]
    )
    assert mod.contains_nonfinite(
        [np.array([1.0, np.inf])]
    )
