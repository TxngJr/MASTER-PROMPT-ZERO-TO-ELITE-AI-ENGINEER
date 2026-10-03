from pathlib import Path
import importlib.util

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "analyze_dataset.py"
SPEC = importlib.util.spec_from_file_location("analyze_dataset", MODULE_PATH)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def test_normalize_missing_strings() -> None:
    df = pd.DataFrame(
        {
            "name": ["Ann", " null ", "Bob"],
            "score": [10.0, 20.0, None],
        }
    )

    cleaned = analyzer.normalize_missing_strings(df)

    assert cleaned["name"].isna().sum() == 1
    assert cleaned["score"].isna().sum() == 1


def test_dataset_summary() -> None:
    df = pd.DataFrame(
        {
            "score": [10.0, 20.0, 30.0],
            "group": ["a", "a", "b"],
        }
    )

    summary = analyzer.dataset_summary(df)

    assert summary["rows"] == 3
    assert summary["columns"] == 2
    assert summary["numeric"]["score"]["mean"] == pytest.approx(20.0)
    assert summary["numeric"]["score"]["std_population"] == pytest.approx(
        8.16496580927726
    )


def test_load_dataset_rejects_non_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("x\n1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        analyzer.load_dataset(path)
