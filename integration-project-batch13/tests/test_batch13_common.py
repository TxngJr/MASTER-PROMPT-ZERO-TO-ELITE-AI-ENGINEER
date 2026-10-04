from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "graph_rl_generative_lab.py"
SPEC = importlib.util.spec_from_file_location("batch13_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_community_graph_shapes() -> None:
    x, a, y = lab.make_community_graph(
        nodes_per_class=10,
        classes=3,
        feature_dim=4,
        seed=3,
    )
    assert x.shape == (30, 4)
    assert a.shape == (30, 30)
    assert y.shape == (30,)
    np.testing.assert_allclose(a, a.T)


def test_chain_environment_reaches_goal() -> None:
    env = lab.ChainEnv(length=5)
    env.reset()
    done = False
    reward = 0.0

    for _ in range(4):
        _, reward, done = env.step(1)

    assert done
    assert reward == 1.0


def test_generative_report_is_finite() -> None:
    report = lab.run_generative(seed=7)

    assert np.isfinite(report["single_gaussian_mean_nll"])
    assert len(report["data_mean"]) == 2
    assert len(report["forward_noising"]) == 4
