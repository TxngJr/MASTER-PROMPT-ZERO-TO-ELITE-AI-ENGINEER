from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "gan_math.py"
SPEC = importlib.util.spec_from_file_location("gan_math_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_bce_with_logits_is_stable() -> None:
    logits = np.array([-1000.0, 1000.0])
    targets = np.array([0.0, 1.0])

    loss = mod.bce_with_logits(logits, targets)
    assert np.isfinite(loss)
    assert loss < 1e-8


def test_good_discriminator_has_lower_loss() -> None:
    good = mod.discriminator_loss(
        np.array([5.0, 4.0]),
        np.array([-5.0, -4.0]),
    )
    bad = mod.discriminator_loss(
        np.array([-5.0, -4.0]),
        np.array([5.0, 4.0]),
    )

    assert good < bad


def test_non_saturating_generator_prefers_real_like_logits() -> None:
    good = mod.generator_non_saturating_loss(np.array([4.0, 5.0]))
    bad = mod.generator_non_saturating_loss(np.array([-4.0, -5.0]))
    assert good < bad
