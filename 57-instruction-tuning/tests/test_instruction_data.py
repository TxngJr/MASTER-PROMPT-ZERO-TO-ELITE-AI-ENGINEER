from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "instruction_data.py"
SPEC = importlib.util.spec_from_file_location("instruction_data_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_render_chat_roles() -> None:
    text = mod.render_simple_chat(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert "hello" in text
    assert "hi" in text


def test_assistant_only_mask() -> None:
    ids, mask = mod.concatenate_role_segments(
        [
            ("system", [1, 2]),
            ("user", [3, 4]),
            ("assistant", [5, 6, 7]),
        ]
    )
    labels = mod.assistant_only_labels(ids, mask)

    assert labels == [-100, -100, -100, -100, 5, 6, 7]
    assert mod.supervised_token_count(labels) == 3


def test_truncate_rejects_zero_supervision() -> None:
    try:
        mod.truncate_example(
            [1, 2, 3, 4],
            [-100, -100, 3, 4],
            max_length=2,
            keep="left",
        )
    except ValueError as exc:
        assert "all supervised" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_truncate_keeps_assistant_tail() -> None:
    ids, labels = mod.truncate_example(
        [1, 2, 3, 4],
        [-100, -100, 3, 4],
        max_length=2,
        keep="right",
    )
    assert ids == [3, 4]
    assert labels == [3, 4]


def test_exact_dedup() -> None:
    example = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ]
    kept, removed = mod.exact_deduplicate_examples(
        [example, example]
    )
    assert len(kept) == 1
    assert removed == [1]
