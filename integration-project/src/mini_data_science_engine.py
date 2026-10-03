"""Batch 01 integration project: CSV -> validated statistical report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MISSING_STRINGS = {"", "na", "n/a", "null", "none", "missing"}


def manual_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("manual_mean requires at least one value")
    return sum(values) / len(values)


def manual_population_variance(values: list[float]) -> float:
    if not values:
        raise ValueError("manual_population_variance requires values")
    center = manual_mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input CSV not found: {path}")
    if not path.is_file():
        raise ValueError(f"input path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("input must be a .csv file")
    return pd.read_csv(path)


def normalize_missing_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        text = cleaned[column].astype("string").str.strip()
        missing_mask = text.str.lower().isin(MISSING_STRINGS)
        cleaned[column] = text.mask(missing_mask, pd.NA)
    return cleaned


def numeric_statistics(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}

    for column in df.select_dtypes(include=[np.number]).columns:
        array = df[column].dropna().to_numpy(dtype=float)
        if array.size == 0:
            continue

        values = array.tolist()
        mean_python = manual_mean(values)
        variance_python = manual_population_variance(values)
        mean_numpy = float(np.mean(array))
        variance_numpy = float(np.var(array, ddof=0))

        if not math.isclose(mean_python, mean_numpy, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"mean implementations disagree for {column}")
        if not math.isclose(
            variance_python,
            variance_numpy,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"variance implementations disagree for {column}")

        result[column] = {
            "count": int(array.size),
            "mean": mean_numpy,
            "variance_population": variance_numpy,
            "std_population": float(np.std(array, ddof=0)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    return result


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    correlation = (
        numeric.corr().round(6).to_dict()
        if numeric.shape[1] >= 2
        else {}
    )

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_statistics": numeric_statistics(df),
        "correlation": correlation,
    }


def save_histograms(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for column in df.select_dtypes(include=[np.number]).columns:
        values = df[column].dropna()
        if values.empty:
            continue

        fig, ax = plt.subplots()
        ax.hist(values, bins="auto")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        fig.tight_layout()

        output = plot_dir / f"{column}_hist.png"
        fig.savefig(output, dpi=120)
        plt.close(fig)
        written.append(output)

    return written


def render_markdown(
    *,
    input_path: Path,
    analysis: dict[str, Any],
    plot_paths: list[Path],
) -> str:
    missing_lines = "\n".join(
        f"- `{column}`: {count}"
        for column, count in analysis["missing"].items()
    )

    numeric_lines: list[str] = []
    for column, stats in analysis["numeric_statistics"].items():
        numeric_lines.append(
            f"- `{column}`: count={stats['count']}, "
            f"mean={stats['mean']:.4f}, "
            f"std={stats['std_population']:.4f}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}"
        )

    plot_lines = "\n".join(f"- `{path.name}`" for path in plot_paths)

    return (
        "# Mini Data Science Engine Report\n\n"
        f"Input: `{input_path}`\n\n"
        f"Rows: **{analysis['rows']}**  \n"
        f"Columns: **{analysis['columns']}**  \n"
        f"Duplicate rows: **{analysis['duplicate_rows']}**\n\n"
        "## Missing values\n\n"
        f"{missing_lines or '- none'}\n\n"
        "## Numeric statistics\n\n"
        f"{chr(10).join(numeric_lines) or '- no numeric columns'}\n\n"
        "## Correlation\n\n"
        "See `report.json` for the full numeric correlation table. "
        "Correlation is descriptive and does not prove causation.\n\n"
        "## Generated plots\n\n"
        f"{plot_lines or '- none'}\n"
    )


def run_pipeline(input_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = load_csv(input_path)
    cleaned = normalize_missing_strings(raw)
    analysis = analyze(cleaned)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = save_histograms(cleaned, output_dir)

    json_path = output_dir / "report.json"
    json_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown_path = output_dir / "report.md"
    markdown_path.write_text(
        render_markdown(
            input_path=input_path,
            analysis=analysis,
            plot_paths=plot_paths,
        ),
        encoding="utf-8",
    )

    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a reproducible Batch 01 data report."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/integration"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analysis = run_pipeline(args.csv_path, args.output_dir)
    print(
        f"completed: {analysis['rows']} rows, "
        f"{analysis['columns']} columns -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
