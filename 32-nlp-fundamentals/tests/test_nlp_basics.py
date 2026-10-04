from pathlib import Path
import importlib.util
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "nlp_basics.py"
SPEC = importlib.util.spec_from_file_location("nlp_basics_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_unicode_normalization_equivalence() -> None:
    composed = "é"
    decomposed = "e\u0301"
    assert mod.normalize_unicode(composed, "NFC") == mod.normalize_unicode(
        decomposed,
        "NFC",
    )


def test_vocabulary_unknown_policy() -> None:
    vocab = mod.Vocabulary.build(
        [["hello", "world"], ["hello"]],
        min_frequency=2,
    )
    encoded = vocab.encode(["hello", "world"])

    assert encoded[0] != vocab.token_to_id["<UNK>"]
    assert encoded[1] == vocab.token_to_id["<UNK>"]


def test_bigram_probability_sums_over_training_vocab() -> None:
    tokens = "a b a c a b".split()
    model = mod.NGramLanguageModel(n=2, smoothing=1.0).fit(tokens)

    total = sum(
        model.probability(token, ("a",))
        for token in model.vocabulary_
    )
    np.testing.assert_allclose(total, 1.0)


def test_perplexity_is_finite() -> None:
    tokens = "a b a b a b a b".split()
    model = mod.NGramLanguageModel(n=2, smoothing=0.5).fit(tokens)
    ppl = model.perplexity(tokens)

    assert np.isfinite(ppl)
    assert ppl >= 1.0


def test_next_token_windows() -> None:
    x, y = mod.make_next_token_windows([1, 2, 3, 4, 5], 3)

    np.testing.assert_array_equal(x, [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(y, [[2, 3, 4], [3, 4, 5]])
