from pathlib import Path
import importlib.util

MODULE_PATH = Path(__file__).parents[1] / "src" / "final_gate.py"
SPEC = importlib.util.spec_from_file_location("final_gate", MODULE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_final_framework_independent_gate() -> None:
    result = gate.run_framework_independent_gate()
    assert result["pass"]
    assert result["metadata_complete"]
    assert result["seed_count"] == 3
