from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "fine_tune_lora_sft_lab.py"
SPEC = importlib.util.spec_from_file_location("batch19_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_adaptation_problem_is_low_rank_shift() -> None:
    problem = lab.make_low_rank_adaptation_problem(
        samples=16,
        input_dim=8,
        output_dim=5,
        rank=2,
        seed=4,
    )

    assert problem["x"].shape == (16, 8)
    assert problem["base_weight"].shape == (5, 8)
    assert problem["target_weight"].shape == (5, 8)


def test_instruction_examples_have_masked_prompts() -> None:
    examples = lab.build_instruction_examples()

    assert len(examples) == 4

    for example in examples:
        labels = example["labels"]
        assert labels[:4] == [-100, -100, -100, -100]
        assert all(label != -100 for label in labels[4:])
