from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "anomaly.py"
SPEC = importlib.util.spec_from_file_location("anomaly_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_zscore_flags_extreme_point() -> None:
    train = np.array([[0.0], [0.1], [-0.1], [0.05], [-0.05]])
    model = mod.ZScoreDetector().fit(train)
    scores = model.score_samples(np.array([[0.0], [10.0]]))
    assert scores[1] > scores[0] * 20


def test_robust_detector_less_distorted_by_extreme_training_value() -> None:
    train = np.array([[0.0], [0.1], [-0.1], [0.05], [20.0]])
    z = mod.ZScoreDetector().fit(train)
    robust = mod.RobustZScoreDetector().fit(train)

    query = np.array([[2.0]])
    assert robust.score_samples(query)[0] > z.score_samples(query)[0]


def test_mahalanobis_captures_correlated_direction() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    train = np.column_stack([x, x + rng.normal(0, 0.05, size=500)])

    model = mod.MahalanobisDetector().fit(train)
    on_manifold = np.array([[3.0, 3.0]])
    off_manifold = np.array([[3.0, -3.0]])

    assert model.score_samples(off_manifold)[0] > model.score_samples(on_manifold)[0]


def test_quantile_threshold_predicts_binary() -> None:
    scores = np.arange(100, dtype=float)
    threshold = mod.threshold_from_quantile(scores, contamination=0.1)
    pred = mod.predict_from_scores(scores, threshold)
    assert 9 <= pred.sum() <= 11
