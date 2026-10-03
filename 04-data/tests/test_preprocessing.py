from pathlib import Path
import importlib.util

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "preprocessing.py"
SPEC = importlib.util.spec_from_file_location("preprocessing", MODULE_PATH)
assert SPEC and SPEC.loader
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def test_split_indices_are_disjoint_and_complete() -> None:
    train, val, test = prep.split_indices(20, seed=7)
    combined = np.concatenate([train, val, test])

    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert sorted(combined.tolist()) == list(range(20))


def test_standardizer_fits_training_statistics() -> None:
    X_train = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    scaler = prep.Standardizer().fit(X_train)

    transformed = scaler.transform(X_train)

    assert transformed[:, 0].mean() == pytest.approx(0.0)
    assert transformed[:, 0].std(ddof=0) == pytest.approx(1.0)
    assert np.all(transformed[:, 1] == 0.0)


def test_standardizer_uses_same_train_state_for_test() -> None:
    X_train = np.array([[0.0], [2.0]])
    X_test = np.array([[10.0]])

    scaler = prep.Standardizer().fit(X_train)

    assert scaler.transform(X_test)[0, 0] == pytest.approx(9.0)


def test_median_imputer() -> None:
    X_train = np.array([[1.0, 10.0], [3.0, np.nan], [100.0, 30.0]])
    imputer = prep.MedianImputer().fit(X_train)
    result = imputer.transform(np.array([[np.nan, np.nan]]))

    assert result[0, 0] == 3.0
    assert result[0, 1] == 20.0


def test_one_hot_unknown_category_is_all_zero_for_feature() -> None:
    encoder = prep.SimpleOneHotEncoder().fit([["red"], ["blue"]])
    encoded = encoder.transform([["green"]])

    assert encoded.shape == (1, 2)
    assert encoded.sum() == 0.0
