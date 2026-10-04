from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "generative_math.py"
SPEC = importlib.util.spec_from_file_location("generative_math_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_gaussian_nll_prefers_correct_mean() -> None:
    x = np.array([1.0, 1.0, 1.0])
    good = mod.gaussian_nll(x, np.ones(3), 0.0)
    bad = mod.gaussian_nll(x, np.zeros(3), 0.0)
    assert good < bad


def test_categorical_nll_is_finite_for_large_logits() -> None:
    loss = mod.categorical_nll(
        np.array([1000.0, 999.0, -1000.0]),
        0,
    )
    assert np.isfinite(loss)
    assert loss < 1.0


def test_forward_diffusion_endpoints() -> None:
    x = np.array([1.0, 2.0, 3.0])

    noisy_clean, _ = mod.forward_diffusion_sample(
        x,
        1.0,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_allclose(noisy_clean, x)

    noisy_noise, noise = mod.forward_diffusion_sample(
        x,
        0.0,
        rng=np.random.default_rng(2),
    )
    np.testing.assert_allclose(noisy_noise, noise)


def test_lower_energy_has_higher_probability() -> None:
    probs = mod.energy_to_probability(np.array([0.0, 2.0, 4.0]))
    assert probs[0] > probs[1] > probs[2]
    np.testing.assert_allclose(probs.sum(), 1.0)


def test_effective_sample_size_uniform() -> None:
    ess = mod.effective_sample_size(np.ones(5) / 5)
    np.testing.assert_allclose(ess, 5.0)
