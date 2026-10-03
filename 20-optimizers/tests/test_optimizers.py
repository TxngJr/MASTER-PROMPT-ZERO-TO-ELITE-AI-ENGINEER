from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "optimizers.py"
SPEC = importlib.util.spec_from_file_location("optimizers_course", MODULE_PATH)
assert SPEC and SPEC.loader
opt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(opt)


class Param:
    def __init__(self, value):
        self.data = np.asarray(value, dtype=float)
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


def optimize_quadratic(optimizer_cls, **kwargs):
    parameter = Param(np.array([5.0]))
    optimizer = optimizer_cls([parameter], **kwargs)
    losses = []

    for _ in range(250):
        losses.append(float(parameter.data[0] ** 2))
        parameter.grad = 2.0 * parameter.data.copy()
        optimizer.step()
        optimizer.zero_grad()

    return parameter.data[0], losses


def test_sgd_reduces_quadratic_loss() -> None:
    value, losses = optimize_quadratic(opt.SGD, lr=0.05)
    assert abs(value) < 1e-6
    assert losses[-1] < losses[0]


def test_momentum_reduces_quadratic_loss() -> None:
    value, losses = optimize_quadratic(
        opt.Momentum,
        lr=0.03,
        momentum=0.9,
    )
    assert abs(value) < 0.05
    assert losses[-1] < losses[0]


def test_adaptive_optimizers_reduce_loss() -> None:
    for cls, kwargs in [
        (opt.AdaGrad, {"lr": 0.5}),
        (opt.RMSProp, {"lr": 0.05}),
        (opt.Adam, {"lr": 0.05}),
        (opt.AdamW, {"lr": 0.05, "weight_decay": 0.0}),
    ]:
        value, losses = optimize_quadratic(cls, **kwargs)
        assert abs(value) < 0.25
        assert losses[-1] < losses[0]


def test_adam_first_step_bias_correction() -> None:
    parameter = Param(np.array([1.0]))
    parameter.grad = np.array([2.0])
    optimizer = opt.Adam(
        [parameter],
        lr=0.1,
        beta1=0.9,
        beta2=0.999,
        eps=1e-12,
    )
    optimizer.step()

    # First-step bias-corrected m_hat=2 and v_hat=4 => update about 0.1.
    assert np.allclose(parameter.data, [0.9], atol=1e-10)


def test_adamw_decays_even_with_zero_gradient() -> None:
    parameter = Param(np.array([2.0]))
    optimizer = opt.AdamW(
        [parameter],
        lr=0.1,
        weight_decay=0.2,
    )
    optimizer.step()

    assert np.allclose(parameter.data, [1.96])


def test_zero_grad_resets_parameter_gradient() -> None:
    parameter = Param(np.array([1.0, 2.0]))
    parameter.grad[:] = 7.0
    optimizer = opt.SGD([parameter], lr=0.1)
    optimizer.zero_grad()
    np.testing.assert_allclose(parameter.grad, [0.0, 0.0])
