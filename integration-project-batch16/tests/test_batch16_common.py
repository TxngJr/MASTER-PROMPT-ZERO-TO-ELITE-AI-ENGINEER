from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "rag_agent_llm_lab.py"
SPEC = importlib.util.spec_from_file_location("batch16_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_rag_retrieves_relevant_source() -> None:
    report = lab.run_rag()

    assert report["top_source_is_relevant"]
    assert report["fused_ranking"][0] == "doc-rmsnorm"
    assert "doc-rmsnorm" in report["packed_source_ids"]


def test_agent_finishes_and_blocks_write() -> None:
    report = lab.run_agent()

    assert report["done"]
    assert report["steps"] == 2
    assert report["tool_calls"] == 1
    assert report["write_blocked_by_default"]
    assert "root mean square" in report["answer"].lower()
