from pathlib import Path
import importlib.util

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "capstone_utils.py"
SPEC = importlib.util.spec_from_file_location("capstone_utils", MODULE_PATH)
assert SPEC and SPEC.loader
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def test_causal_windows_shift() -> None:
    x, y = utils.causal_windows([1, 2, 3, 4, 5], sequence_length=4)
    assert x.tolist() == [[1, 2, 3, 4]]
    assert y.tolist() == [[2, 3, 4, 5]]


def test_memory_estimates() -> None:
    assert utils.model_weight_bytes(10, bytes_per_parameter=4) == 40
    assert utils.adam_training_state_bytes(10) == 160
    assert utils.kv_cache_bytes(
        layers=2, batch_size=1, sequence_length=8,
        kv_heads=4, head_dim=16, bytes_per_element=2,
    ) == 4096


def test_int8_roundtrip_is_close() -> None:
    values = np.array([-2.0, -0.1, 0.0, 0.4, 1.7], dtype=np.float32)
    q, scale = utils.symmetric_int8_quantize(values)
    restored = utils.symmetric_int8_dequantize(q, scale)
    assert q.dtype == np.int8
    assert np.max(np.abs(values - restored)) <= scale / 2 + 1e-6


def test_dpo_loss_prefers_positive_margin() -> None:
    better = utils.dpo_loss_from_log_ratios(
        policy_chosen_logp=-1, policy_rejected_logp=-3,
        reference_chosen_logp=-2, reference_rejected_logp=-3, beta=1.0,
    )
    worse = utils.dpo_loss_from_log_ratios(
        policy_chosen_logp=-3, policy_rejected_logp=-1,
        reference_chosen_logp=-2, reference_rejected_logp=-3, beta=1.0,
    )
    assert better < worse


def test_manifest_fingerprint_order_independent() -> None:
    assert utils.manifest_fingerprint({"a": 1, "b": 2}) == utils.manifest_fingerprint(
        {"b": 2, "a": 1}
    )


def test_invalid_kv_dimensions() -> None:
    with pytest.raises(ValueError):
        utils.kv_cache_bytes(
            layers=0, batch_size=1, sequence_length=8,
            kv_heads=4, head_dim=16, bytes_per_element=2,
        )
