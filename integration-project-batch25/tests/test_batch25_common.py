from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "reasoning_vlm_voice_lab.py"
SPEC = importlib.util.spec_from_file_location("batch25_lab", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_reasoning_mode() -> None:
    report = lab.run_reasoning()

    assert report["majority_answer"] == "42"
    assert report["best_of_n_answer"] == "42"
    assert report["self_consistency"] == 0.6
    assert sum(report["sample_token_allocations"]) == 2000


def test_vlm_mode() -> None:
    report = lab.run_vlm()

    assert report["remaining_context_tokens"] > 0
    assert report["first_box_iou"] == 1.0
    assert report["grounding"]["precision"] == 0.5
    assert report["grounding"]["recall"] == 1.0


def test_voice_mode() -> None:
    report = lab.run_voice()

    assert report["endpoint_frame"] == 5
    assert report["rtf"] == 0.25
    assert report["codec_bitrate_kbps"] == 4.0
    assert report["first_response_ms"] == 440.0


def test_full_gate() -> None:
    report = lab.run_full()
    assert report["system_ready"]
