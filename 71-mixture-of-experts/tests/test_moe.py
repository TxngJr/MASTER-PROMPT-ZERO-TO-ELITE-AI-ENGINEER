from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "moe.py"
SPEC = importlib.util.spec_from_file_location("moe_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_top_k_router_normalizes_selected_weights() -> None:
    logits = np.array(
        [
            [3.0, 2.0, 1.0],
            [0.0, 4.0, 2.0],
        ]
    )
    indices, weights, probabilities = mod.top_k_router(
        logits,
        top_k=2,
    )

    assert indices.shape == (2, 2)
    assert weights.shape == (2, 2)
    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(
        weights.sum(axis=1),
        1.0,
    )


def test_expert_capacity() -> None:
    assert mod.expert_capacity(
        tokens=16,
        num_experts=4,
        top_k=2,
        capacity_factor=1.25,
    ) == 10


def test_capacity_keeps_highest_weight_assignments() -> None:
    indices = np.array(
        [
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    weights = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
        ]
    )
    accepted = mod.apply_capacity(
        indices,
        weights,
        num_experts=2,
        capacity=2,
    )

    assert accepted[:, 0].tolist() == [True, True, False]
    assert accepted[:, 1].tolist() == [False, True, True]


def test_uniform_router_has_unit_switch_aux_loss() -> None:
    probabilities = np.full((4, 2), 0.5)
    assignments = np.array([0, 1, 0, 1])

    loss = mod.switch_load_balance_loss(
        probabilities,
        assignments,
    )
    np.testing.assert_allclose(loss, 1.0)


def test_active_parameter_fraction() -> None:
    fraction = mod.active_parameter_fraction(
        shared_parameters=20,
        expert_parameters_each=10,
        num_experts=8,
        active_experts=2,
    )
    np.testing.assert_allclose(fraction, 40 / 100)
