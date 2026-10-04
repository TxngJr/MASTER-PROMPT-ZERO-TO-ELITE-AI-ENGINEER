from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "world_model.py"
SPEC = importlib.util.spec_from_file_location("world_model_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_fit_linear_dynamics_recovers_system() -> None:
    rng = np.random.default_rng(2)
    a_true = np.array([[1.0, 0.1], [0.0, 0.95]])
    b_true = np.array([[0.0], [0.2]])

    states = rng.normal(size=(400, 2))
    actions = rng.normal(size=(400, 1))
    next_states = states @ a_true.T + actions @ b_true.T

    a, b = mod.fit_linear_dynamics(
        states,
        actions,
        next_states,
        l2=1e-10,
    )

    np.testing.assert_allclose(a, a_true, atol=1e-8)
    np.testing.assert_allclose(b, b_true, atol=1e-8)


def test_rollout_linear_dynamics() -> None:
    a = np.eye(1)
    b = np.ones((1, 1))
    actions = np.array([[1.0], [2.0], [-1.0]])

    trajectory = mod.rollout_linear_dynamics(
        np.array([0.0]),
        actions,
        transition_matrix=a,
        action_matrix=b,
    )

    np.testing.assert_allclose(
        trajectory[:, 0],
        [0.0, 1.0, 3.0, 2.0],
    )


def test_discounted_return() -> None:
    value = mod.discounted_return(
        np.array([1.0, 1.0, 1.0]),
        gamma=0.5,
    )
    np.testing.assert_allclose(value, 1.75)


def test_planner_moves_toward_goal() -> None:
    plans = mod.enumerate_action_plans(
        [-1.0, 0.0, 1.0],
        horizon=2,
    )
    plan, score = mod.plan_by_model(
        np.array([0.0]),
        plans,
        transition_matrix=np.eye(1),
        action_matrix=np.ones((1, 1)),
        goal=np.array([2.0]),
        action_penalty=0.01,
    )

    np.testing.assert_allclose(
        plan,
        [[1.0], [1.0]],
    )
    assert np.isfinite(score)


def test_jepa_loss_zero_when_embeddings_match() -> None:
    embeddings = np.arange(12, dtype=float).reshape(4, 3)
    loss = mod.jepa_prediction_loss(
        embeddings,
        embeddings.copy(),
    )
    assert loss == 0.0
