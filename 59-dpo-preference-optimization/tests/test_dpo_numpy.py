from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "dpo_numpy.py"
SPEC = importlib.util.spec_from_file_location("dpo_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_dpo_loss_improves_when_policy_favors_chosen() -> None:
    reference_chosen = np.array([-2.0])
    reference_rejected = np.array([-2.0])

    neutral = mod.dpo_loss(
        np.array([-2.0]),
        np.array([-2.0]),
        reference_chosen,
        reference_rejected,
        beta=1.0,
    )
    improved = mod.dpo_loss(
        np.array([-1.0]),
        np.array([-3.0]),
        reference_chosen,
        reference_rejected,
        beta=1.0,
    )

    assert improved < neutral


def test_policy_reference_cancellation() -> None:
    logits = mod.dpo_logits(
        np.array([-1.0]),
        np.array([-3.0]),
        np.array([-1.0]),
        np.array([-3.0]),
        beta=0.2,
    )
    np.testing.assert_allclose(logits, 0.0)


def test_completion_logprob_respects_mask() -> None:
    token_log_probs = np.array(
        [[-1.0, -2.0, -3.0, -4.0]]
    )
    mask = np.array(
        [[False, False, True, True]]
    )
    score = mod.completion_logprob(
        token_log_probs,
        mask,
    )
    np.testing.assert_allclose(score, [-7.0])


def test_length_audit() -> None:
    report = mod.length_audit(
        np.array([10, 20, 30]),
        np.array([8, 22, 25]),
    )
    np.testing.assert_allclose(report["chosen_mean"], 20.0)
    assert 0.0 <= report["chosen_longer_fraction"] <= 1.0


def test_ipo_loss_zero_at_target_gap() -> None:
    beta = 0.5
    target_gap = 1.0 / (2.0 * beta)

    loss = mod.ipo_squared_loss(
        np.array([target_gap]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        beta=beta,
    )
    np.testing.assert_allclose(loss, 0.0)
