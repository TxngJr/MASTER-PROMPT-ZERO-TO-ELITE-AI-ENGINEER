"""Bounding-box and segmentation metrics implemented with NumPy."""

from __future__ import annotations

import numpy as np


def box_area(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("boxes must have shape (N,4) in xyxy format")

    width = np.maximum(0.0, values[:, 2] - values[:, 0])
    height = np.maximum(0.0, values[:, 3] - values[:, 1])
    return width * height


def pairwise_iou(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    a = np.asarray(boxes1, dtype=float)
    b = np.asarray(boxes2, dtype=float)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("boxes must be matrices")
    if a.shape[1:] != (4,) or b.shape[1:] != (4,):
        raise ValueError("boxes must have shape (N,4)/(M,4)")

    area_a = box_area(a)
    area_b = box_area(b)

    top_left = np.maximum(
        a[:, None, :2],
        b[None, :, :2],
    )
    bottom_right = np.minimum(
        a[:, None, 2:],
        b[None, :, 2:],
    )

    size = np.maximum(0.0, bottom_right - top_left)
    intersection = size[..., 0] * size[..., 1]

    union = (
        area_a[:, None]
        + area_b[None, :]
        - intersection
    )

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def box_iou(
    box1: np.ndarray,
    box2: np.ndarray,
) -> float:
    return float(
        pairwise_iou(
            np.asarray(box1, dtype=float).reshape(1, 4),
            np.asarray(box2, dtype=float).reshape(1, 4),
        )[0, 0]
    )


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    values = np.asarray(boxes, dtype=float)
    confidence = np.asarray(scores, dtype=float).reshape(-1)

    if values.shape != (len(confidence), 4):
        raise ValueError("box/score shape mismatch")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in [0,1]")

    order = np.argsort(-confidence, kind="stable")
    keep: list[int] = []

    while len(order) > 0:
        current = int(order[0])
        keep.append(current)

        if len(order) == 1:
            break

        remaining = order[1:]
        overlaps = pairwise_iou(
            values[current : current + 1],
            values[remaining],
        )[0]

        order = remaining[overlaps <= iou_threshold]

    return keep


def mask_iou(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)

    if pred.shape != truth.shape:
        raise ValueError("mask shapes must match")

    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)


def dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)

    if pred.shape != truth.shape:
        raise ValueError("mask shapes must match")

    intersection = np.logical_and(pred, truth).sum()
    denominator = pred.sum() + truth.sum()

    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)
