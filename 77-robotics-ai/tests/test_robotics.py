from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "robotics.py"
SPEC = importlib.util.spec_from_file_location("robotics_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_planar_two_link_fk() -> None:
    x, y = mod.planar_two_link_fk(
        0.0,
        0.0,
        link1=1.0,
        link2=1.0,
    )
    np.testing.assert_allclose([x, y], [2.0, 0.0])


def test_clamp_action() -> None:
    clamped = mod.clamp_action(
        np.array([2.0, -3.0]),
        low=np.array([-1.0, -2.0]),
        high=np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(clamped, [1.0, -2.0])


def test_proportional_control() -> None:
    action = mod.proportional_control(
        np.array([1.0, -1.0]),
        np.array([0.0, 0.0]),
        gain=0.5,
        max_abs_action=0.4,
    )
    np.testing.assert_allclose(action, [0.4, -0.4])


def test_behavior_cloning_mse() -> None:
    loss = mod.behavior_cloning_mse(
        np.array([[1.0, 2.0], [2.0, 3.0]]),
        np.array([[1.0, 1.0], [2.0, 2.0]]),
    )
    np.testing.assert_allclose(loss, 0.5)


def test_action_chunk() -> None:
    actions = np.arange(12, dtype=float).reshape(6, 2)
    chunk = mod.action_chunk(
        actions,
        start=2,
        chunk_size=3,
    )
    np.testing.assert_allclose(
        chunk,
        actions[2:5],
    )


def test_latency_budget() -> None:
    assert mod.latency_budget_ok(
        observation_ms=3,
        inference_ms=10,
        action_dispatch_ms=2,
        control_frequency_hz=50,
    )
    assert not mod.latency_budget_ok(
        observation_ms=5,
        inference_ms=20,
        action_dispatch_ms=5,
        control_frequency_hz=50,
    )
