from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "rlhf_numpy.py"
SPEC = importlib.util.spec_from_file_location("rlhf_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_bradley_terry_prefers_larger_reward() -> None:
    probability = mod.bradley_terry_probability(
        np.array([3.0]),
        np.array([1.0]),
    )
    assert probability[0] > 0.5


def test_reward_loss_decreases_with_better_margin() -> None:
    bad = mod.reward_model_loss(
        np.array([0.0]),
        np.array([1.0]),
    )
    good = mod.reward_model_loss(
        np.array([3.0]),
        np.array([1.0]),
    )
    assert good < bad


def test_categorical_kl_is_nonnegative() -> None:
    kl = mod.categorical_kl(
        np.array([[2.0, 0.0, -1.0]]),
        np.array([[1.0, 1.0, 0.0]]),
    )
    assert kl.shape == (1,)
    assert kl[0] >= -1e-12


def test_sampled_log_ratio_can_be_negative() -> None:
    value = mod.sampled_log_ratio(
        np.array([-2.0]),
        np.array([-1.0]),
    )
    assert value[0] < 0


def test_advantages_are_normalized() -> None:
    values = mod.normalize_advantages(
        np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(np.mean(values), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(values), 1.0, atol=1e-12)


def test_ppo_clipping_limits_positive_advantage_gain() -> None:
    old = np.log(np.array([0.5]))
    new = np.log(np.array([0.9]))
    advantage = np.array([1.0])

    loss = mod.ppo_clipped_policy_loss(
        new,
        old,
        advantage,
        clip_range=0.2,
    )

    np.testing.assert_allclose(loss, -1.2)
