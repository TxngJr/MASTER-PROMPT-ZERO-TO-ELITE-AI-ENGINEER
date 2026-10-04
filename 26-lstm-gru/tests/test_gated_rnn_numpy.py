from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "gated_rnn_numpy.py"
SPEC = importlib.util.spec_from_file_location("gated_rnn_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_lstm_shapes_and_gate_ranges() -> None:
    cell = mod.LSTMCellNumPy(3, 4, seed=2)
    x = np.ones((5, 3))
    h = np.zeros((5, 4))
    c = np.zeros((5, 4))

    h_next, c_next, gates = cell.step(x, h, c)

    assert h_next.shape == (5, 4)
    assert c_next.shape == (5, 4)
    assert set(gates) == {"forget", "input", "candidate", "output"}

    for name in ["forget", "input", "output"]:
        assert np.all(gates[name] >= 0.0)
        assert np.all(gates[name] <= 1.0)


def test_lstm_sequence_shapes() -> None:
    cell = mod.LSTMCellNumPy(2, 6, seed=7)
    x = np.ones((4, 8, 2))

    output, h, c = cell.forward(x)

    assert output.shape == (4, 8, 6)
    assert h.shape == (4, 6)
    assert c.shape == (4, 6)


def test_gru_shapes_and_gate_ranges() -> None:
    cell = mod.GRUCellNumPy(3, 5, seed=4)
    x = np.ones((2, 3))
    h = np.zeros((2, 5))

    h_next, gates = cell.step(x, h)

    assert h_next.shape == (2, 5)
    assert np.all((gates["reset"] >= 0.0) & (gates["reset"] <= 1.0))
    assert np.all((gates["update"] >= 0.0) & (gates["update"] <= 1.0))


def test_gru_sequence_final_matches_last_output() -> None:
    cell = mod.GRUCellNumPy(1, 3, seed=5)
    x = np.arange(10, dtype=float).reshape(2, 5, 1)

    output, final_hidden = cell.forward(x)

    np.testing.assert_allclose(output[:, -1, :], final_hidden)
