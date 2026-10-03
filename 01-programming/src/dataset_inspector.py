"""Small CSV inspector using only the Python standard library."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MISSING_TOKENS = {"", "na", "n/a", "null", "none"}


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_TOKENS


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header row")
        return [dict(row) for row in reader]


def infer_column_type(values: list[str]) -> str:
    observed = [value.strip() for value in values if not is_missing(value)]
    if not observed:
        return "unknown"

    try:
        for value in observed:
            int(value)
        return "int"
    except ValueError:
        pass

    try:
        for value in observed:
            float(value)
        return "float"
    except ValueError:
        return "str"


def numeric_summary(values: list[str]) -> dict[str, float]:
    numbers = [float(value) for value in values if not is_missing(value)]
    if not numbers:
        return {}
    return {
        "min": min(numbers),
        "max": max(numbers),
        "mean": sum(numbers) / len(numbers),
    }


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "columns": 0, "schema": {}}

    columns = list(rows[0].keys())
    schema: dict[str, Any] = {}

    for column in columns:
        values = [row.get(column, "") for row in rows]
        inferred = infer_column_type(values)
        item: dict[str, Any] = {
            "type": inferred,
            "missing": sum(is_missing(value) for value in values),
            "unique": len({value for value in values if not is_missing(value)}),
        }
        if inferred in {"int", "float"}:
            item.update(numeric_summary(values))
        schema[column] = item

    return {
        "rows": len(rows),
        "columns": len(columns),
        "schema": schema,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a CSV dataset.")
    parser.add_argument("csv_path", type=Path, help="Path to a CSV file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_csv(args.csv_path)
    summary = summarize_rows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
