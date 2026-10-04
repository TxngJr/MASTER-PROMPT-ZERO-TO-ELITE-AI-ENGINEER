from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "rlhf_dpo_eval_lab.py"
SPEC = importlib.util.spec_from_file_location("batch20_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_preference_problem_shapes() -> None:
    problem = lab.make_preference_problem(
        num_prompts=5,
        num_responses=3,
        feature_dim=4,
        seed=3,
    )

    assert problem["response_features"].shape == (5, 3, 4)
    assert problem["true_utilities"].shape == (5, 3)
    assert len(problem["chosen"]) == 5
    assert len(problem["rejected"]) == 5


def test_evaluation_report_is_reproducible() -> None:
    first = lab.run_evaluation(seed=7)
    second = lab.run_evaluation(seed=7)

    assert first == second
    assert first["manifest"]["paired_comparison"]
    assert first["contamination_rate"] == 0.5
    assert 0.0 <= first["pairwise_rates"]["a_win_rate"] <= 1.0
