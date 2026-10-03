from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "dataset_inspector.py"
SPEC = importlib.util.spec_from_file_location("dataset_inspector", MODULE_PATH)
assert SPEC and SPEC.loader
dataset_inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dataset_inspector)


def test_infer_column_type() -> None:
    assert dataset_inspector.infer_column_type(["1", "2", ""]) == "int"
    assert dataset_inspector.infer_column_type(["1.5", "2.0"]) == "float"
    assert dataset_inspector.infer_column_type(["cat", "dog"]) == "str"


def test_load_and_summarize(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "name,score,age\n"
        "Ann,10.5,20\n"
        "Bob,,21\n"
        "Cara,7.5,20\n",
        encoding="utf-8",
    )

    rows = dataset_inspector.load_csv(csv_path)
    summary = dataset_inspector.summarize_rows(rows)

    assert summary["rows"] == 3
    assert summary["columns"] == 3
    assert summary["schema"]["score"]["missing"] == 1
    assert summary["schema"]["score"]["mean"] == 9.0
    assert summary["schema"]["age"]["type"] == "int"


def test_empty_rows() -> None:
    assert dataset_inspector.summarize_rows([]) == {
        "rows": 0,
        "columns": 0,
        "schema": {},
    }
