from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "diffusion_numpy.py"
SPEC = importlib.util.spec_from_file_location("diffusion_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_schedule_monotonic() -> None:
    betas = mod.linear_beta_schedule(10)
    assert np.all(np.diff(betas) > 0)
    alphas, bars = mod.alpha_terms(betas)
    assert np.all(alphas < 1.0)
    assert np.all(np.diff(bars) < 0)


def test_q_sample_endpoints() -> None:
    x = np.array([1.0, 2.0])
    noise = np.array([3.0, 4.0])

    clean, _ = mod.q_sample(
        x,
        1.0,
        rng=np.random.default_rng(1),
        noise=noise,
    )
    np.testing.assert_allclose(clean, x)

    pure_noise, _ = mod.q_sample(
        x,
        0.0,
        rng=np.random.default_rng(1),
        noise=noise,
    )
    np.testing.assert_allclose(pure_noise, noise)


def test_predict_x0_inverts_forward_equation() -> None:
    rng = np.random.default_rng(2)
    x0 = rng.normal(size=(4, 3))
    xt, eps = mod.q_sample(x0, 0.6, rng=rng)
    recovered = mod.predict_x0_from_epsilon(xt, eps, 0.6)
    np.testing.assert_allclose(recovered, x0)


def test_posterior_variance_nonnegative() -> None:
    value = mod.posterior_variance(
        beta_t=0.01,
        alpha_bar_t=0.7,
        alpha_bar_previous=0.75,
    )
    assert value >= 0.0


def test_reverse_mean_is_finite() -> None:
    mean = mod.ddpm_reverse_mean(
        np.ones(3),
        np.zeros(3),
        beta_t=0.01,
        alpha_t=0.99,
        alpha_bar_t=0.8,
    )
    assert np.all(np.isfinite(mean))
