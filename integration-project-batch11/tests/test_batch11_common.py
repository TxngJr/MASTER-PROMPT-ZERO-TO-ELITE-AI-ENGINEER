from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "vision_nlp_lab.py"
SPEC = importlib.util.spec_from_file_location("batch11_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_embedding_lab_is_finite() -> None:
    report = lab.run_embedding_lab(
        epochs=1,
        embedding_dim=6,
        seed=3,
    )

    assert report["vocabulary_size"] > 1
    assert report["pair_count"] > 0
    assert np.all(np.isfinite(report["skipgram_loss_history"]))
    assert np.all(np.isfinite(report["glove_loss_history"]))
    assert np.isfinite(report["fasttext_oov_norm"])
