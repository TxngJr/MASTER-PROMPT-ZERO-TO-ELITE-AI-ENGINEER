from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "llm_numpy.py"
SPEC = importlib.util.spec_from_file_location("llm_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_rms_norm_unit_rms_without_affine() -> None:
    x = np.array([[3.0, 4.0, 0.0, 0.0]])
    out = mod.rms_norm(x, eps=1e-12)
    rms = np.sqrt(np.mean(out**2, axis=-1))
    np.testing.assert_allclose(rms, 1.0, atol=1e-6)


def test_rope_position_zero_is_identity() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 2, 1, 4))
    cos, sin = mod.rope_angles(1, 4, start_position=0)
    rotated = mod.apply_rope(x, cos, sin)
    np.testing.assert_allclose(rotated, x)


def test_repeat_kv_expands_heads() -> None:
    x = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
    repeated = mod.repeat_kv(x, query_heads=8)
    assert repeated.shape == (1, 8, 3, 4)
    np.testing.assert_array_equal(repeated[:, 0], repeated[:, 1])


def test_causal_attention_cannot_use_future() -> None:
    q = np.ones((1, 1, 3, 2))
    k = np.ones((1, 1, 3, 2))
    v = np.array(
        [[[[1.0, 0.0], [10.0, 0.0], [100.0, 0.0]]]]
    )

    output, weights = mod.causal_attention(q, k, v)

    np.testing.assert_allclose(weights[0, 0, 0, 1:], 0.0)
    np.testing.assert_allclose(output[0, 0, 0], [1.0, 0.0])


def test_swiglu_shape() -> None:
    gate = np.array([[1.0, -1.0, 0.5]])
    value = np.ones_like(gate)
    result = mod.swiglu(gate, value)
    assert result.shape == gate.shape
    assert np.all(np.isfinite(result))


def test_gqa_cache_uses_fewer_bytes_than_mha() -> None:
    mha = mod.kv_cache_bytes(
        layers=32,
        batch_size=1,
        kv_heads=32,
        sequence_length=2048,
        head_dim=128,
        bytes_per_element=2,
    )
    gqa = mod.kv_cache_bytes(
        layers=32,
        batch_size=1,
        kv_heads=8,
        sequence_length=2048,
        head_dim=128,
        bytes_per_element=2,
    )
    assert gqa == mha // 4
