from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "rnn_numpy.py"
SPEC = importlib.util.spec_from_file_location("rnn_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_forward_shapes() -> None:
    model = mod.TanhRNN(3, 5, output_size=2, seed=7)
    x = np.ones((4, 6, 3))

    hidden_sequence, final_hidden = model.forward(x)
    logits = model.readout(hidden_sequence)

    assert hidden_sequence.shape == (4, 6, 5)
    assert final_hidden.shape == (4, 5)
    assert logits.shape == (4, 6, 2)


def test_manual_recurrence_matches_forward() -> None:
    model = mod.TanhRNN(2, 3, seed=3)
    x = np.array(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ]
    )

    output, final_hidden = model.forward(x)

    h0 = np.zeros((1, 3))
    h1 = model.step(x[:, 0, :], h0)
    h2 = model.step(x[:, 1, :], h1)

    np.testing.assert_allclose(output[:, 0, :], h1)
    np.testing.assert_allclose(output[:, 1, :], h2)
    np.testing.assert_allclose(final_hidden, h2)


def test_initial_hidden_changes_sequence() -> None:
    model = mod.TanhRNN(1, 2, seed=5)
    x = np.zeros((1, 3, 1))

    a, _ = model.forward(x)
    b, _ = model.forward(x, h0=np.ones((1, 2)))

    assert not np.allclose(a, b)
