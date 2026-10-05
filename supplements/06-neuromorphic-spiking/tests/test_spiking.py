from pathlib import Path
import importlib.util
import numpy as np

P=Path(__file__).parents[1]/"src"/"spiking.py"
S=importlib.util.spec_from_file_location("spiking",P)
m=importlib.util.module_from_spec(S)
S.loader.exec_module(m)


def test_lif_spikes_and_resets():
    currents=np.array([[0.6],[0.6],[0.6],[0.6]])
    v,s=m.simulate_lif(currents,beta=1-1e-9,threshold=1.0,reset=0.0)
    assert s[:,0].sum() == 2
    assert np.all(v[s[:,0].astype(bool),0] == 0.0)


def test_spike_rate():
    rate=m.spike_rate([[0,1],[1,1],[0,0],[1,0]])
    assert np.allclose(rate,[0.5,0.5])
