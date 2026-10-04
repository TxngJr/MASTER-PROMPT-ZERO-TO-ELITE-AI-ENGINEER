from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "quantization_numpy.py"
SPEC = importlib.util.spec_from_file_location("quantization_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_symmetric_roundtrip_has_small_error() -> None:
    x = np.linspace(-1.0, 1.0, 101)
    q, scale = mod.symmetric_quantize(x, bits=8)
    restored = mod.symmetric_dequantize(q, scale=scale)

    assert mod.quantization_mse(x, restored) < 1e-4


def test_asymmetric_handles_positive_range() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    q, scale, zero = mod.asymmetric_quantize(x, bits=8)
    restored = mod.asymmetric_dequantize(
        q,
        scale=scale,
        zero_point=zero,
    )

    assert mod.quantization_mse(x, restored) < 1e-4


def test_per_channel_can_outperform_single_scale() -> None:
    x = np.array(
        [
            [0.01, -0.01, 0.02],
            [10.0, -10.0, 5.0],
        ]
    )

    q_global, scale_global = mod.symmetric_quantize(
        x,
        bits=4,
    )
    global_restored = mod.symmetric_dequantize(
        q_global,
        scale=scale_global,
    )

    q_channel, scales = mod.per_channel_symmetric_quantize(
        x,
        axis=0,
        bits=4,
    )
    channel_restored = np.vstack(
        [
            mod.symmetric_dequantize(
                q_channel[index],
                scale=scales[index],
            )
            for index in range(len(scales))
        ]
    )

    assert (
        mod.quantization_mse(x, channel_restored)
        <= mod.quantization_mse(x, global_restored)
    )


def test_grouped_quantization() -> None:
    x = np.arange(10, dtype=float)
    q, scales = mod.grouped_symmetric_quantize(
        x,
        group_size=4,
        bits=4,
    )

    assert q.shape == (10,)
    assert scales.shape == (3,)


def test_storage_math() -> None:
    assert mod.ideal_storage_bytes(
        1_000_000,
        bits_per_value=4,
    ) == 500_000
    assert mod.compression_ratio(16, 4) == 4.0
