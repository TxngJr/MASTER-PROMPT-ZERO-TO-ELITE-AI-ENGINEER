"""Vision-language token, projection, and grounding utilities."""

from __future__ import annotations

import math
import numpy as np


def patch_grid(
    image_height: int,
    image_width: int,
    *,
    patch_height: int,
    patch_width: int | None = None,
) -> tuple[int, int]:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if patch_height <= 0:
        raise ValueError("patch_height must be positive")

    width = patch_height if patch_width is None else patch_width
    if width <= 0:
        raise ValueError("patch_width must be positive")

    rows = math.ceil(image_height / patch_height)
    cols = math.ceil(image_width / width)
    return int(rows), int(cols)


def image_token_count(
    image_height: int,
    image_width: int,
    *,
    patch_height: int,
    patch_width: int | None = None,
    extra_tokens: int = 0,
) -> int:
    if extra_tokens < 0:
        raise ValueError("extra_tokens must be non-negative")

    rows, cols = patch_grid(
        image_height,
        image_width,
        patch_height=patch_height,
        patch_width=patch_width,
    )
    return int(rows * cols + extra_tokens)


def linear_project(
    vision_tokens: np.ndarray,
    weight: np.ndarray,
    *,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(vision_tokens, dtype=float)
    w = np.asarray(weight, dtype=float)

    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("vision_tokens and weight must be 2D")
    if x.shape[1] != w.shape[0]:
        raise ValueError("projection dimension mismatch")

    output = x @ w

    if bias is not None:
        b = np.asarray(bias, dtype=float)
        if b.shape != (w.shape[1],):
            raise ValueError("bias shape mismatch")
        output = output + b

    return output


def multimodal_context_remaining(
    *,
    max_context_tokens: int,
    text_tokens: int,
    image_tokens: list[int],
    reserved_output_tokens: int = 0,
) -> int:
    values = [
        max_context_tokens,
        text_tokens,
        reserved_output_tokens,
        *image_tokens,
    ]
    if any(value < 0 for value in values):
        raise ValueError("token counts must be non-negative")

    used = text_tokens + sum(image_tokens) + reserved_output_tokens
    return int(max_context_tokens - used)


def normalized_box_to_pixels(
    box_xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    x1, y1, x2, y2 = box_xyxy
    if not (
        0.0 <= x1 <= x2 <= 1.0
        and 0.0 <= y1 <= y2 <= 1.0
    ):
        raise ValueError("normalized box must satisfy 0<=x1<=x2<=1")

    return (
        x1 * image_width,
        y1 * image_height,
        x2 * image_width,
        y2 * image_height,
    )


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    if ax2 < ax1 or ay2 < ay1 or bx2 < bx1 or by2 < by1:
        raise ValueError("invalid box coordinates")

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def grounding_counts(
    predicted_boxes: list[tuple[float, float, float, float]],
    target_boxes: list[tuple[float, float, float, float]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in [0,1]")

    matched_targets: set[int] = set()
    true_positive = 0

    for predicted in predicted_boxes:
        best_target = None
        best_iou = -1.0

        for index, target in enumerate(target_boxes):
            if index in matched_targets:
                continue

            score = bbox_iou(predicted, target)
            if score > best_iou:
                best_iou = score
                best_target = index

        if (
            best_target is not None
            and best_iou >= iou_threshold
        ):
            matched_targets.add(best_target)
            true_positive += 1

    false_positive = len(predicted_boxes) - true_positive
    false_negative = len(target_boxes) - true_positive

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def grounding_precision_recall(
    predicted_boxes: list[tuple[float, float, float, float]],
    target_boxes: list[tuple[float, float, float, float]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    counts = grounding_counts(
        predicted_boxes,
        target_boxes,
        iou_threshold=iou_threshold,
    )
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
    }
