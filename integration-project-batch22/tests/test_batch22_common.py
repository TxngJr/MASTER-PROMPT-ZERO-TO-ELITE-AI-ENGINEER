from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "production_ml_lifecycle_lab.py"
SPEC = importlib.util.spec_from_file_location("batch22_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_data_pipeline() -> None:
    report = lab.run_data_pipeline()

    assert report["validation_errors"] == []
    assert report["input_records"] == 3
    assert report["output_records"] == 2
    assert len(report["lineage_fingerprint"]) == 64


def test_mlops_promotion() -> None:
    report = lab.run_mlops_promotion()

    assert report["promoted"]
    assert report["aliases"]["champion"] == "v4"
    assert report["psi"] >= 0.0


def test_deployment_plan() -> None:
    report = lab.run_deployment_plan()

    assert report["desired_replicas"] == 6
    assert report["rollout"]["max_total_pods"] == 7
    assert report["canary_requests"] == 500
    assert report["fits_gpu_pool"]


def test_full_lifecycle_gate() -> None:
    report = lab.run_full_lifecycle()

    assert report["ready_for_rollout"]
    assert report["data"]["validation_errors"] == []
    assert report["mlops"]["promoted"]
    assert report["deployment"]["fits_gpu_pool"]
