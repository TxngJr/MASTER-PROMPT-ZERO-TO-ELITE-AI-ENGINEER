from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional


MODULE_PATH = Path(__file__).parents[1] / "src" / "attention_numpy.py"
SPEC = importlib.util.spec_from_file_location("attention_numpy_torch_compare", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_numpy_attention_matches_pytorch_sdpa_without_dropout() -> None:
    rng = np.random.default_rng(11)
    q = rng.normal(size=(2, 3, 5, 4))
    k = rng.normal(size=(2, 3, 5, 4))
    v = rng.normal(size=(2, 3, 5, 6))

    ours, _ = mod.scaled_dot_product_attention(q, k, v)

    reference = F.scaled_dot_product_attention(
        torch.tensor(q, dtype=torch.float64),
        torch.tensor(k, dtype=torch.float64),
        torch.tensor(v, dtype=torch.float64),
        dropout_p=0.0,
    ).numpy()

    np.testing.assert_allclose(ours, reference, rtol=1e-10, atol=1e-10)
