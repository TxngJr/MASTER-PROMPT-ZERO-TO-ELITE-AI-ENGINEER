from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "diffusion_multimodal_speech_lab.py"
SPEC = importlib.util.spec_from_file_location("batch14_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_four_mode_points() -> None:
    points = lab.make_four_mode_points(200, seed=3)
    assert points.shape == (200, 2)
    assert np.all(np.isfinite(points))


def test_paired_modalities_shapes() -> None:
    image, text = lab.make_paired_modalities(
        100,
        latent_dim=4,
        observed_dim=7,
        seed=4,
    )
    assert image.shape == (100, 7)
    assert text.shape == (100, 7)


def test_synthetic_tones_finite() -> None:
    waveform = lab.synthesize_token_tones(
        [1, 2, 3],
        sample_rate=8000,
    )
    assert waveform.ndim == 1
    assert len(waveform) > 0
    assert np.all(np.isfinite(waveform))
