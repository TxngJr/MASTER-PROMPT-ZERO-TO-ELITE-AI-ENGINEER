from pathlib import Path
import importlib.util
import sys

import numpy as np
from sklearn.decomposition import PCA


MODULE_PATH = Path(__file__).parents[1] / "src" / "pca.py"
SPEC = importlib.util.spec_from_file_location("pca_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_pca_matches_sklearn_explained_variance() -> None:
    rng = np.random.default_rng(9)
    base = rng.normal(size=(300, 2))
    X = np.column_stack(
        [
            base[:, 0],
            2.0 * base[:, 0] + 0.1 * base[:, 1],
            base[:, 1],
        ]
    )

    ours = mod.PCAFromScratch(n_components=2).fit(X)
    ref = PCA(n_components=2, svd_solver="full").fit(X)

    np.testing.assert_allclose(
        ours.explained_variance_,
        ref.explained_variance_,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        ours.explained_variance_ratio_,
        ref.explained_variance_ratio_,
        rtol=1e-10,
        atol=1e-10,
    )


def test_full_components_reconstruct_data() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 4))

    model = mod.PCAFromScratch(n_components=4)
    Z = model.fit_transform(X)
    reconstructed = model.inverse_transform(Z)

    assert mod.reconstruction_mse(X, reconstructed) < 1e-20


def test_fewer_components_have_nonzero_reconstruction_error() -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(120, 5))

    model = mod.PCAFromScratch(n_components=2)
    reconstructed = model.inverse_transform(model.fit_transform(X))

    assert mod.reconstruction_mse(X, reconstructed) > 0.0
    assert model.explained_variance_ratio_.sum() < 1.0
