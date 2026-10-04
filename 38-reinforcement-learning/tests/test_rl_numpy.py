from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "rl_numpy.py"
SPEC = importlib.util.spec_from_file_location("rl_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_discounted_returns() -> None:
    result = mod.discounted_returns([1.0, 1.0, 1.0], gamma=0.5)
    np.testing.assert_allclose(result, [1.75, 1.5, 1.0])


def test_terminal_bellman_does_not_bootstrap() -> None:
    target = mod.bellman_optimality_backup(
        2.0,
        np.array([100.0, 200.0]),
        gamma=0.99,
        done=True,
    )
    assert target == 2.0


def test_q_learning_update_moves_toward_target() -> None:
    updated, td = mod.q_learning_update(
        0.0,
        1.0,
        np.array([2.0, 3.0]),
        alpha=0.5,
        gamma=0.9,
        done=False,
    )
    assert td > 0.0
    assert updated > 0.0


def test_epsilon_zero_is_greedy() -> None:
    action = mod.epsilon_greedy(
        np.array([1.0, 4.0, 2.0]),
        epsilon=0.0,
        rng=np.random.default_rng(3),
    )
    assert action == 1


def test_ppo_clipping_positive_advantage() -> None:
    old = np.log(np.array([0.5]))
    new = np.log(np.array([0.9]))
    adv = np.array([2.0])

    surrogate = mod.ppo_clipped_surrogate(
        new,
        old,
        adv,
        clip_epsilon=0.2,
    )
    np.testing.assert_allclose(surrogate, [2.4])
