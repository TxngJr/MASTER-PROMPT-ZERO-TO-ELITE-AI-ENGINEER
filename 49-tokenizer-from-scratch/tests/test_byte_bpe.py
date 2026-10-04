from pathlib import Path
import importlib.util
import sys


MODULE_PATH = Path(__file__).parents[1] / "src" / "byte_bpe.py"
SPEC = importlib.util.spec_from_file_location("byte_bpe_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_merge_pair_non_overlapping() -> None:
    assert mod.merge_pair(
        [1, 2, 1, 2, 3],
        (1, 2),
        256,
    ) == [256, 256, 3]


def test_training_is_deterministic() -> None:
    corpus = ["abababab", "abab", "abba"]
    first = mod.train_bpe(corpus, num_merges=6)
    second = mod.train_bpe(corpus, num_merges=6)
    assert first == second


def test_roundtrip_ascii_and_thai() -> None:
    corpus = [
        "hello hello world",
        "ภาษาไทย ภาษาไทย",
    ]
    merges = mod.train_bpe(corpus, num_merges=20)
    tokenizer = mod.ByteBPETokenizer(merges)

    for text in corpus + ["hello ภาษาไทย"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_learned_merges_reduce_repetitive_sequence() -> None:
    text = "banana banana banana"
    tokenizer = mod.ByteBPETokenizer(
        mod.train_bpe([text], num_merges=10)
    )
    encoded = tokenizer.encode(text)

    assert len(encoded) < len(text.encode("utf-8"))
    assert tokenizer.bytes_per_token(text) > 1.0


def test_unknown_token_id_rejected() -> None:
    tokenizer = mod.ByteBPETokenizer([])
    try:
        tokenizer.decode([999])
    except ValueError as exc:
        assert "unknown token id" in str(exc)
    else:
        raise AssertionError("expected ValueError")
