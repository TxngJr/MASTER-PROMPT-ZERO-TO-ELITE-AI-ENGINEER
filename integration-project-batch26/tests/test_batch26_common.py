from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "world_robot_edge_lab.py"
SPEC = importlib.util.spec_from_file_location("batch26_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_world_mode() -> None:
    report = lab.run_world_model()
    assert report["plan"] == [[1.0], [1.0], [1.0]]
    assert report["final_state"] == 3.0


def test_robotics_mode() -> None:
    report = lab.run_robotics()
    assert report["control_latency_ok"]
    assert report["safe_action"] == [0.5, -0.5]
    assert report["control_period_ms"] == 20.0


def test_edge_mode() -> None:
    report = lab.run_edge()
    assert report["parameter_compression_ratio"] == 4.0
    assert report["fits_64k_budget"]
    assert report["operator_coverage"]["coverage"] == 1.0
    assert report["quantization_max_abs_error"] >= 0.0


def test_full_gate() -> None:
    report = lab.run_full()
    assert report["system_ready"]
