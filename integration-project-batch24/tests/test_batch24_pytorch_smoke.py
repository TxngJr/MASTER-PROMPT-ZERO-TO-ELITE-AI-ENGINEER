from pathlib import Path
import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
prune = pytest.importorskip("torch.nn.utils.prune")


MODULE_PATH = (
    Path(__file__).parents[2]
    / "70-model-compression"
    / "src"
    / "compression.py"
)
SPEC = importlib.util.spec_from_file_location(
    "batch24_compression_torch_compare",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
compression = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compression)


def test_torch_unstructured_pruning_and_remove() -> None:
    layer = torch.nn.Linear(10, 4, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(
                40,
                dtype=torch.float32,
            ).reshape(4, 10)
        )

    prune.l1_unstructured(
        layer,
        name="weight",
        amount=0.5,
    )

    masked = layer.weight.detach()
    zero_fraction = float(
        (masked == 0).float().mean()
    )
    assert zero_fraction == 0.5

    prune.remove(layer, "weight")
    assert hasattr(layer, "weight")
    assert not hasattr(layer, "weight_orig")
    assert float(
        (layer.weight.detach() == 0).float().mean()
    ) == 0.5


def test_torch_kd_matches_numpy_definition() -> None:
    teacher = torch.tensor(
        [[2.0, 1.0, -1.0], [0.5, 2.0, 1.0]],
        dtype=torch.float64,
    )
    student = torch.tensor(
        [[1.8, 1.1, -0.8], [0.7, 1.7, 1.1]],
        dtype=torch.float64,
    )
    temperature = 2.5

    torch_loss = (
        F.kl_div(
            F.log_softmax(
                student / temperature,
                dim=-1,
            ),
            F.softmax(
                teacher / temperature,
                dim=-1,
            ),
            reduction="batchmean",
        )
        * temperature**2
    )

    numpy_loss = compression.distillation_loss(
        teacher.numpy(),
        student.numpy(),
        temperature=temperature,
    )

    np.testing.assert_allclose(
        float(torch_loss),
        numpy_loss,
        rtol=1e-10,
        atol=1e-10,
    )
