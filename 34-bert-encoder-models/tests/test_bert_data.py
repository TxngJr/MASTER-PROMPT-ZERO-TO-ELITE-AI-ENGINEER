from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "bert_data.py"
SPEC = importlib.util.spec_from_file_location("bert_data_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_special_tokens_and_segments() -> None:
    tokens = mod.add_special_tokens(
        [10, 11],
        cls_id=1,
        sep_id=2,
        second=[20, 21, 22],
    )
    segments = mod.create_token_type_ids(
        2,
        second_length=3,
    )

    assert tokens == [1, 10, 11, 2, 20, 21, 22, 2]
    assert segments == [0, 0, 0, 0, 1, 1, 1, 1]


def test_mlm_corrupt_never_selects_special_tokens() -> None:
    tokens = np.array([1, 5, 6, 7, 2])
    corrupted, labels, selected = mod.mlm_corrupt(
        tokens,
        vocab_size=20,
        mask_token_id=3,
        special_token_ids={1, 2, 3, 0},
        selection_probability=1.0,
        rng=np.random.default_rng(7),
    )

    assert not selected[0]
    assert not selected[-1]
    assert labels[0] == -100
    assert labels[-1] == -100
    np.testing.assert_array_equal(labels[1:4], tokens[1:4])
    assert corrupted.shape == tokens.shape


def test_mlm_is_reproducible() -> None:
    tokens = np.arange(20)
    a = mod.mlm_corrupt(
        tokens,
        vocab_size=100,
        mask_token_id=99,
        special_token_ids={0},
        rng=np.random.default_rng(4),
    )
    b = mod.mlm_corrupt(
        tokens,
        vocab_size=100,
        mask_token_id=99,
        special_token_ids={0},
        rng=np.random.default_rng(4),
    )

    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_masked_accuracy() -> None:
    pred = np.array([9, 2, 3, 4])
    labels = np.array([-100, 2, 8, 4])
    assert mod.masked_accuracy(pred, labels) == 2 / 3
