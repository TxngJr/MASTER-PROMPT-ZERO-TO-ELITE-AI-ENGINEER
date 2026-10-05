"""Framework-independent final integration gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _load(relative_path: str, name: str):
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_framework_independent_gate() -> dict[str, Any]:
    paper = _load(
        "79-research-paper-engineering/src/paper_engineering.py",
        "batch27_paper",
    )
    capstone = _load(
        "80-build-llm-from-scratch/src/capstone_utils.py",
        "batch27_utils",
    )

    stats = paper.seed_statistics([0.71, 0.73, 0.72])
    metadata = {
        field: "recorded"
        for field in paper.REQUIRED_REPRODUCIBILITY_FIELDS
    }
    audit = paper.audit_reproducibility_metadata(metadata)

    original = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    q, scale = capstone.symmetric_int8_quantize(original)
    restored = capstone.symmetric_int8_dequantize(q, scale)
    max_error = float(np.max(np.abs(original - restored)))

    passed = bool(
        stats["count"] == 3
        and audit["complete"]
        and len(q) == 3
        and np.isfinite(max_error)
    )

    return {
        "seed_count": stats["count"],
        "mean_metric": stats["mean"],
        "metadata_complete": audit["complete"],
        "quantization_max_error": max_error,
        "pass": passed,
    }


if __name__ == "__main__":
    print(run_framework_independent_gate())
