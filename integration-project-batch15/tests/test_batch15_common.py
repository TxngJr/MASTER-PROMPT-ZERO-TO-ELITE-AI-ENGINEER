from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "recommendation_vision_retrieval_lab.py"
SPEC = importlib.util.spec_from_file_location("batch15_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_recommender_problem_shapes() -> None:
    users, items, positives = lab.make_recommender_problem(
        users=12,
        items=20,
        latent_dim=4,
        positives_per_user=3,
        seed=3,
    )
    assert users.shape == (12, 4)
    assert items.shape == (20, 4)
    assert len(positives) == 36


def test_segmentation_dataset_shapes() -> None:
    images, masks = lab.make_segmentation_dataset(
        10,
        image_size=24,
        seed=4,
    )
    assert images.shape == (10, 1, 24, 24)
    assert masks.shape == images.shape
    assert np.all((masks == 0) | (masks == 1))


def test_vector_search_nprobe_recall() -> None:
    report = lab.run_vector_search(seed=5)

    low = report["mean_recall_at_10_nprobe_1"]
    high = report["mean_recall_at_10_nprobe_4"]

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high >= low
