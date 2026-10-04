from pathlib import Path
import importlib.util

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "compression_moe_memory_lab.py"
SPEC = importlib.util.spec_from_file_location("batch24_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_compression_mode() -> None:
    report = lab.run_compression(seed=3)

    assert report["sparsity"] == 0.5
    assert report["pruned_nonzeros"] < report["original_nonzeros"]
    assert np.isfinite(report["distillation_loss"])
    assert report["distillation_loss"] >= 0.0


def test_moe_mode() -> None:
    report = lab.run_moe()

    assert len(report["expert_load"]) == 4
    assert sum(report["expert_load"]) == 16
    assert 0.0 <= report["dropped_assignment_fraction"] <= 1.0
    assert report["load_balance_loss"] > 0.0
    assert 0.0 < report["active_parameter_fraction"] < 1.0


def test_context_mode() -> None:
    report = lab.run_context()

    assert report["sliding_attention_pairs"] < report["dense_attention_pairs"]
    assert report["pair_reduction"] > 2.0
    assert report["sliding_cache_tokens"] == 512
    assert report["memory_recall_at_3"] == 1.0


def test_full_gate() -> None:
    report = lab.run_full()
    assert report["system_ready"]
