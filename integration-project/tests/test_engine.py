from pathlib import Path
import importlib.util

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "mini_data_science_engine.py"
SPEC = importlib.util.spec_from_file_location("mini_data_science_engine", MODULE_PATH)
assert SPEC and SPEC.loader
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)


def test_manual_statistics() -> None:
    values = [1.0, 2.0, 3.0]
    assert engine.manual_mean(values) == 2.0
    assert engine.manual_population_variance(values) == pytest.approx(2 / 3)


def test_analyze_dataframe() -> None:
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0],
            "label": ["a", "b", "c"],
        }
    )
    result = engine.analyze(df)

    assert result["rows"] == 3
    assert result["columns"] == 3
    assert result["numeric_statistics"]["x"]["mean"] == 2.0
    assert result["correlation"]["x"]["y"] == pytest.approx(1.0)


def test_run_pipeline(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text(
        "name,x,y\n"
        "a,1,2\n"
        "b,2,4\n"
        "c,3,6\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"

    result = engine.run_pipeline(input_path, output_dir)

    assert result["rows"] == 3
    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "plots" / "x_hist.png").exists()
