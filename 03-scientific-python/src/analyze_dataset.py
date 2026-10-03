"""Reusable tabular dataset analysis helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MISSING_STRINGS = {"", "na", "n/a", "null", "none", "missing"}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("this chapter analyzer expects a .csv file")
    return pd.read_csv(path)


def normalize_missing_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        normalized = cleaned[column].astype("string").str.strip()
        mask = normalized.str.lower().isin(MISSING_STRINGS)
        cleaned[column] = normalized.mask(mask, pd.NA)
    return cleaned


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    numeric_summary: dict[str, Any] = {}

    for column in numeric.columns:
        values = numeric[column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        numeric_summary[column] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std_population": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing": {key: int(value) for key, value in df.isna().sum().items()},
        "duplicates": int(df.duplicated().sum()),
        "numeric": numeric_summary,
    }


def save_histograms(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
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

        path = output_dir / f"{column}_hist.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a CSV dataset.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ch03"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = normalize_missing_strings(load_dataset(args.csv_path))
    summary = dataset_summary(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plots = save_histograms(df, args.output_dir)

    print(f"summary: {summary_path}")
    print(f"plots: {len(plots)}")


if __name__ == "__main__":
    main()
