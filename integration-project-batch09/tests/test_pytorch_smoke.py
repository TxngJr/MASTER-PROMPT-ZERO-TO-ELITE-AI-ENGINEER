from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "sequence_latent_lab.py"
SPEC = importlib.util.spec_from_file_location("sequence_latent_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_sequence_models_smoke() -> None:
    report = lab.run_sequence_benchmark(
        epochs=1,
        hidden_size=8,
        batch_size=64,
        seed=3,
        limit=300,
    )

    assert report["selected_model"] in {"rnn", "lstm", "gru"}
    for result in report["results"].values():
        assert result["parameters"] > 0
        assert 0.0 <= result["best_validation_accuracy"] <= 1.0
        assert 0.0 <= result["test_accuracy"] <= 1.0


def test_vae_smoke() -> None:
    report = lab.run_vae(
        epochs=1,
        latent_dim=4,
        batch_size=64,
        beta=1.0,
        seed=4,
        limit=400,
    )

    assert report["parameter_count"] > 0
    assert np.isfinite(report["test"]["total"])
    assert np.isfinite(report["test"]["reconstruction"])
    assert np.isfinite(report["test"]["kl"])
    assert 0.0 <= report["prior_samples"]["min"] <= 1.0
    assert 0.0 <= report["prior_samples"]["max"] <= 1.0
