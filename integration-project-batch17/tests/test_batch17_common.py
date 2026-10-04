from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "tokenizer_dataset_pretraining_lab.py"
SPEC = importlib.util.spec_from_file_location("batch17_lab_common", MODULE_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_pipeline_has_no_document_hash_leakage() -> None:
    report = lab.run_pipeline()

    assert report["removed_duplicate_ids"] == ["duplicate-copy"]
    assert report["train_validation_hash_overlap"] == 0
    assert report["train_test_hash_overlap"] == 0
    assert report["roundtrip_ok"]
    assert report["vocab_size"] > 256


def test_expected_split_groups_exist() -> None:
    built = lab.build_dataset()
    counts = built["manifest"]["split_counts"]

    assert counts["train"] >= 1
    assert counts["validation"] >= 1
    assert counts["test"] >= 1
