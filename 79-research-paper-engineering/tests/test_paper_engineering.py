from pathlib import Path
import importlib.util

import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "paper_engineering.py"
SPEC = importlib.util.spec_from_file_location("paper_engineering", MODULE_PATH)
assert SPEC and SPEC.loader
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)


def test_relative_error() -> None:
    assert paper.relative_error(100.0, 95.0) == pytest.approx(0.05)


def test_result_tolerance() -> None:
    assert paper.result_within_tolerance(0.80, 0.79, relative_tolerance=0.02)
    assert not paper.result_within_tolerance(0.80, 0.70, relative_tolerance=0.02)


def test_improvement_recovery() -> None:
    assert paper.normalized_improvement_recovery(
        baseline=70, claimed=80, reproduced=77
    ) == pytest.approx(0.7)


def test_seed_statistics() -> None:
    stats = paper.seed_statistics([1.0, 2.0, 3.0])
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] > 0
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]


def test_fingerprint_is_order_independent() -> None:
    left = paper.experiment_fingerprint({"lr": 1e-3, "seed": 7})
    right = paper.experiment_fingerprint({"seed": 7, "lr": 1e-3})
    assert left == right
    assert len(left) == 64


def test_metadata_audit() -> None:
    metadata = {field: "x" for field in paper.REQUIRED_REPRODUCIBILITY_FIELDS}
    assert paper.audit_reproducibility_metadata(metadata)["complete"]
    metadata.pop("seed")
    result = paper.audit_reproducibility_metadata(metadata)
    assert not result["complete"]
    assert result["missing"] == ["seed"]
