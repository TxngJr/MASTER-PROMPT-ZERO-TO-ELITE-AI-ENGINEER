from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "vae_math.py"
SPEC = importlib.util.spec_from_file_location("vae_math_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_kl_zero_for_standard_normal_posterior() -> None:
    mu = np.zeros((5, 3))
    logvar = np.zeros((5, 3))

    assert mod.kl_standard_normal(mu, logvar) == 0.0


def test_kl_positive_away_from_prior() -> None:
    mu = np.ones((4, 2))
    logvar = np.zeros((4, 2))

    assert mod.kl_standard_normal(mu, logvar) > 0.0


def test_reparameterization_reproducible_with_seed() -> None:
    mu = np.zeros((2, 3))
    logvar = np.zeros((2, 3))

    a = mod.reparameterize(
        mu,
        logvar,
        rng=np.random.default_rng(7),
    )
    b = mod.reparameterize(
        mu,
        logvar,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(a, b)


def test_bce_logits_is_finite_for_extreme_logits() -> None:
    logits = np.array([[-1000.0, 1000.0]])
    targets = np.array([[0.0, 1.0]])

    loss = mod.binary_cross_entropy_with_logits(logits, targets)
    assert np.isfinite(loss)
    assert loss < 1e-8


def test_vae_loss_components_add_up() -> None:
    logits = np.zeros((3, 4))
    targets = np.ones((3, 4))
    mu = np.ones((3, 2)) * 0.1
    logvar = np.zeros((3, 2))

    result = mod.vae_loss(
        logits,
        targets,
        mu,
        logvar,
        beta=2.0,
    )

    assert result["total"] == (
        result["reconstruction"] + 2.0 * result["kl"]
    )
