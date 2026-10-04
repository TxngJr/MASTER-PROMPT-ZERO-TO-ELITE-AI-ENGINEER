from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "reliable_guarded_xai_lab.py"
SPEC = importlib.util.spec_from_file_location("batch23_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_distributed_mode() -> None:
    report = lab.run_distributed()

    assert report["queue_utilization"] == 0.9
    assert report["admitted"] == 2
    assert report["rejected"] == 3
    assert report["circuit_state"] == "open"


def test_guardrails_mode() -> None:
    report = lab.run_guardrails()

    assert report["visible_document_ids"] == ["course-public"]
    assert report["read_decision"] == "allow"
    assert report["write_without_approval"] == "approval_required"
    assert report["write_with_approval"] == "allow"
    assert report["budget_allows"]


def test_xai_mode() -> None:
    report = lab.run_xai(seed=3)

    assert abs(report["completeness_gap"]) < 1e-6
    assert report["ig_vs_shapley_similarity"] > 0.999
    assert report["permutation_mean"][0] > 0


def test_full_gate() -> None:
    report = lab.run_full()
    assert report["system_ready"]
