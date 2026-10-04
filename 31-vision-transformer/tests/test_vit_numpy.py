from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "vit_numpy.py"
SPEC = importlib.util.spec_from_file_location("vit_numpy_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_patch_count() -> None:
    assert mod.patch_count(8, 8, 2) == 16
    assert mod.patch_count(8, 8, 4) == 4


def test_patchify_known_order() -> None:
    image = np.arange(16).reshape(1, 1, 4, 4)
    patches = mod.patchify_nchw(image, 2)

    expected = np.array(
        [[
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15],
        ]]
    )
    np.testing.assert_array_equal(patches, expected)


def test_patch_embedding_and_cls_shapes() -> None:
    patches = np.ones((3, 4, 4))
    weight = np.ones((4, 6))
    tokens = mod.linear_patch_embedding(patches, weight)

    with_cls = mod.prepend_cls_token(
        tokens,
        np.zeros(6),
    )

    assert tokens.shape == (3, 4, 6)
    assert with_cls.shape == (3, 5, 6)


def test_position_addition() -> None:
    tokens = np.zeros((2, 5, 6))
    position = np.arange(30, dtype=float).reshape(5, 6)

    result = mod.add_learned_positions(tokens, position)

    np.testing.assert_allclose(result[0], position)
    np.testing.assert_allclose(result[1], position)
