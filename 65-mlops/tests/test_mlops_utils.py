from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "mlops_utils.py"
SPEC = importlib.util.spec_from_file_location("mlops_utils_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_fingerprint_is_order_independent_for_config() -> None:
    first = mod.experiment_fingerprint(
        code_revision="abc",
        dataset_revision="data-v1",
        config={"lr": 0.1, "batch": 32},
    )
    second = mod.experiment_fingerprint(
        code_revision="abc",
        dataset_revision="data-v1",
        config={"batch": 32, "lr": 0.1},
    )
    assert first == second


def test_psi_zero_for_same_distribution() -> None:
    score = mod.population_stability_index(
        np.array([10, 20, 30]),
        np.array([10, 20, 30]),
    )
    np.testing.assert_allclose(score, 0.0)


def test_psi_detects_shift() -> None:
    same = mod.population_stability_index(
        np.array([50, 50]),
        np.array([50, 50]),
    )
    shifted = mod.population_stability_index(
        np.array([50, 50]),
        np.array([90, 10]),
    )
    assert shifted > same


def test_promotion_gate() -> None:
    passed, failures = mod.promotion_gate(
        {"accuracy": 0.91, "p99_ms": 400},
        {
            "accuracy": (">=", 0.90),
            "p99_ms": ("<=", 500),
        },
    )
    assert passed
    assert failures == []


def test_alias_is_movable_pointer() -> None:
    aliases = {"champion": "v3"}
    updated = mod.registry_alias_update(
        aliases,
        alias="champion",
        version="v4",
    )
    assert aliases["champion"] == "v3"
    assert updated["champion"] == "v4"


def test_rolling_error_rate() -> None:
    rates = mod.rolling_error_rate(
        [True, False, False, True],
        window=2,
    )
    np.testing.assert_allclose(
        rates,
        [0.0, 0.5, 1.0, 0.5],
    )
