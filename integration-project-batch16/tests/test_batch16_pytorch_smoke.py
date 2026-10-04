from pathlib import Path
import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "rag_agent_llm_lab.py"
SPEC = importlib.util.spec_from_file_location("batch16_lab_torch", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_modern_llm_smoke() -> None:
    report = lab.run_llm(
        steps=1,
        seed=5,
    )

    assert report["parameter_count"] > 0
    assert np.isfinite(report["loss_first"])
    assert np.isfinite(report["loss_last"])
    assert report["query_heads"] == 4
    assert report["kv_heads"] == 2
    assert report["causal_prefix_max_diff"] < 1e-5
    assert report["example_kv_cache_bytes_2048"] > 0
