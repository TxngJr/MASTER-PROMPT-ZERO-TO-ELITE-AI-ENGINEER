from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "vlm.py"
SPEC = importlib.util.spec_from_file_location("vlm_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_patch_grid_ceil_behavior() -> None:
    rows, cols = mod.patch_grid(
        230,
        250,
        patch_height=16,
    )
    assert rows == 15
    assert cols == 16


def test_image_token_count() -> None:
    count = mod.image_token_count(
        224,
        224,
        patch_height=14,
        extra_tokens=1,
    )
    assert count == 257


def test_linear_project_shape() -> None:
    tokens = np.ones((8, 4))
    weight = np.arange(12, dtype=float).reshape(4, 3)
    output = mod.linear_project(tokens, weight)
    assert output.shape == (8, 3)


def test_multimodal_context_remaining() -> None:
    remaining = mod.multimodal_context_remaining(
        max_context_tokens=4096,
        text_tokens=1000,
        image_tokens=[576, 576],
        reserved_output_tokens=512,
    )
    assert remaining == 1432


def test_bbox_iou() -> None:
    score = mod.bbox_iou(
        (0.0, 0.0, 2.0, 2.0),
        (1.0, 1.0, 3.0, 3.0),
    )
    np.testing.assert_allclose(score, 1 / 7)


def test_grounding_metrics() -> None:
    predictions = [
        (0.0, 0.0, 1.0, 1.0),
        (10.0, 10.0, 12.0, 12.0),
    ]
    targets = [
        (0.0, 0.0, 1.0, 1.0),
    ]

    metrics = mod.grounding_precision_recall(
        predictions,
        targets,
        iou_threshold=0.5,
    )
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
