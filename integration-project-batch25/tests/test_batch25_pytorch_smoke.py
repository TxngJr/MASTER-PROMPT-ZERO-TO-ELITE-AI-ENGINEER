from pathlib import Path
import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")


VLM_PATH = (
    Path(__file__).parents[2]
    / "74-vision-language-models"
    / "src"
    / "vlm.py"
)
VOICE_PATH = (
    Path(__file__).parents[2]
    / "75-audio-voice-models"
    / "src"
    / "voice.py"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vlm = _load(VLM_PATH, "batch25_vlm_torch_compare")
voice = _load(VOICE_PATH, "batch25_voice_torch_compare")


def test_torch_patchification_matches_patch_count() -> None:
    image = torch.randn(1, 3, 224, 224)
    unfold = torch.nn.Unfold(
        kernel_size=(14, 14),
        stride=(14, 14),
    )
    patches = unfold(image)

    expected = vlm.image_token_count(
        224,
        224,
        patch_height=14,
    )
    assert patches.shape[-1] == expected


def test_torch_linear_projection_matches_numpy() -> None:
    rng = np.random.default_rng(4)
    tokens = rng.normal(size=(6, 4))
    weight = rng.normal(size=(4, 3))
    bias = rng.normal(size=(3,))

    numpy_output = vlm.linear_project(
        tokens,
        weight,
        bias=bias,
    )

    layer = torch.nn.Linear(
        4,
        3,
        bias=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(weight.T, dtype=torch.float64)
        )
        layer.bias.copy_(
            torch.tensor(bias, dtype=torch.float64)
        )

    torch_output = layer(
        torch.tensor(tokens, dtype=torch.float64)
    )

    np.testing.assert_allclose(
        torch_output.detach().numpy(),
        numpy_output,
        rtol=1e-12,
        atol=1e-12,
    )


def test_torch_rms_matches_numpy() -> None:
    waveform = np.array(
        [0.5, -0.25, 0.75, -1.0],
        dtype=np.float64,
    )
    numpy_rms = voice.rms_energy(waveform)

    tensor = torch.tensor(
        waveform,
        dtype=torch.float64,
    )
    torch_rms = torch.sqrt(torch.mean(tensor**2))

    np.testing.assert_allclose(
        float(torch_rms),
        numpy_rms,
        rtol=1e-12,
        atol=1e-12,
    )
