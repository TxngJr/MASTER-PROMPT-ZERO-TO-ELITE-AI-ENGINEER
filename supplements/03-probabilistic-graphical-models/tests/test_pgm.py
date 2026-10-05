from pathlib import Path
import importlib.util
import numpy as np

P = Path(__file__).parents[1] / "src" / "pgm.py"
S = importlib.util.spec_from_file_location("pgm", P)
m = importlib.util.module_from_spec(S)
S.loader.exec_module(m)


def test_bayes_update():
    posterior = m.bayes_update([0.5, 0.5], [0.8, 0.2])
    assert np.allclose(posterior, [0.8, 0.2])


def test_hmm_forward():
    initial = [0.6, 0.4]
    transition = [[0.7, 0.3], [0.2, 0.8]]
    emissions = [[0.5, 0.1], [0.4, 0.6]]
    hist, likelihood = m.hmm_forward(initial, transition, emissions)
    assert hist.shape == (2, 2)
    assert 0 < likelihood <= 1
