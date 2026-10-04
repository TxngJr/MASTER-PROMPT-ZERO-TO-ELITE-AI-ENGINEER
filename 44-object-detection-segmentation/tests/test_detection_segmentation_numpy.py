from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "detection_segmentation_numpy.py"
SPEC = importlib.util.spec_from_file_location("detection_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_box_iou_identical_is_one() -> None:
    box = np.array([0.0, 0.0, 2.0, 2.0])
    np.testing.assert_allclose(mod.box_iou(box, box), 1.0)


def test_pairwise_iou_shape() -> None:
    a = np.array([[0, 0, 2, 2], [2, 2, 4, 4]], dtype=float)
    b = np.array([[0, 0, 1, 1]], dtype=float)

    matrix = mod.pairwise_iou(a, b)
    assert matrix.shape == (2, 1)
    assert matrix[0, 0] > matrix[1, 0]


def test_nms_keeps_high_score_and_far_box() -> None:
    boxes = np.array(
        [
            [0, 0, 2, 2],
            [0.1, 0.1, 2.1, 2.1],
            [5, 5, 7, 7],
        ],
        dtype=float,
    )
    scores = np.array([0.9, 0.8, 0.7])

    keep = mod.nms(boxes, scores, 0.5)

    assert keep == [0, 2]


def test_mask_metrics() -> None:
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [1, 0]], dtype=bool)

    np.testing.assert_allclose(mod.mask_iou(prediction, target), 1 / 3)
    np.testing.assert_allclose(mod.dice_score(prediction, target), 0.5)
