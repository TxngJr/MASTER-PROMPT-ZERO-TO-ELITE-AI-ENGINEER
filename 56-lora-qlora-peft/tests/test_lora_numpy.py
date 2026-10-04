from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "lora_numpy.py"
SPEC = importlib.util.spec_from_file_location("lora_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_lora_parameter_count_is_smaller() -> None:
    full = 4096 * 4096
    adapter = mod.lora_parameter_count(4096, 4096, 8)

    assert adapter == 65536
    assert adapter < full


def test_zero_b_is_initial_noop() -> None:
    rng = np.random.default_rng(3)
    weight = rng.normal(size=(4, 5))
    a = rng.normal(size=(2, 5))
    b = np.zeros((4, 2))
    x = rng.normal(size=(7, 5))

    base = x @ weight.T
    adapted = mod.lora_forward(
        x,
        weight,
        a,
        b,
        alpha=4.0,
    )

    np.testing.assert_allclose(adapted, base)


def test_merge_unmerge_roundtrip() -> None:
    rng = np.random.default_rng(4)
    weight = rng.normal(size=(4, 5))
    a = rng.normal(size=(2, 5))
    b = rng.normal(size=(4, 2))

    merged = mod.merge_lora(
        weight,
        a,
        b,
        alpha=8.0,
    )
    recovered = mod.unmerge_lora(
        merged,
        a,
        b,
        alpha=8.0,
    )

    np.testing.assert_allclose(recovered, weight)


def test_trainable_fraction() -> None:
    fraction = mod.trainable_fraction(
        base_parameters=1_000_000,
        adapter_parameters=10_000,
    )
    np.testing.assert_allclose(
        fraction,
        10_000 / 1_010_000,
    )


def test_generic_codebook_quantizer() -> None:
    values = np.array([-1.0, -0.2, 0.6, 1.0])
    codebook = np.array([-1.0, 0.0, 1.0])

    quantized, indices = mod.nearest_codebook_quantize(
        values,
        codebook,
    )

    np.testing.assert_allclose(
        quantized,
        [-1.0, 0.0, 1.0, 1.0],
    )
    assert indices.dtype == np.int64
