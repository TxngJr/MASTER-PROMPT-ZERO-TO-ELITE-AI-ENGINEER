from pathlib import Path
import importlib.util

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "seq2seq_data.py"
SPEC = importlib.util.spec_from_file_location("seq2seq_data_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_shift_right() -> None:
    assert mod.shift_right(
        [10, 11, 12, 2],
        start_token_id=1,
    ) == [1, 10, 11, 12]


def test_padding_mask() -> None:
    tokens = np.array([[1, 2, 0], [3, 0, 0]])
    mask = mod.make_padding_mask(tokens, pad_token_id=0)
    np.testing.assert_array_equal(
        mask,
        [[True, True, False], [True, False, False]],
    )


def test_span_corruption() -> None:
    source, target = mod.span_corrupt(
        [10, 11, 12, 13, 14, 15, 16],
        [(2, 4), (5, 6)],
        sentinel_ids=[90, 91],
        eos_token_id=2,
    )

    assert source == [10, 11, 90, 14, 91, 16]
    assert target == [90, 12, 13, 91, 15, 2]


def test_span_corruption_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        mod.span_corrupt(
            [1, 2, 3, 4],
            [(1, 3), (2, 4)],
            sentinel_ids=[90, 91],
            eos_token_id=2,
        )
