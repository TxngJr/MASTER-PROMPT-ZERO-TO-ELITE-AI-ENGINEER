import numpy as np
import pytest

torch = pytest.importorskip("torch")
captum = pytest.importorskip("captum.attr")

from captum.attr import IntegratedGradients


def test_captum_integrated_gradients_linear_completeness() -> None:
    model = torch.nn.Linear(3, 1, bias=True)

    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[2.0, -3.0, 0.5]])
        )
        model.bias.fill_(7.0)

    model.eval()
    x = torch.tensor(
        [[1.0, 2.0, 3.0]],
        requires_grad=True,
    )
    baseline = torch.zeros_like(x)

    ig = IntegratedGradients(model)
    attribution, delta = ig.attribute(
        x,
        baselines=baseline,
        n_steps=64,
        return_convergence_delta=True,
    )

    expected = torch.tensor(
        [[2.0, -6.0, 1.5]]
    )

    assert torch.allclose(
        attribution,
        expected,
        atol=1e-4,
    )
    assert np.isfinite(float(delta.abs().max()))
    assert float(delta.abs().max()) < 1e-4
