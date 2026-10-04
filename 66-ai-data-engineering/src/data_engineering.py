"""Data-contract, event-time, partition and lineage utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any


_TYPE_MAP = {
    "str": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
}


def validate_record(
    record: dict[str, Any],
    schema: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    for field, spec in schema.items():
        nullable = bool(spec.get("nullable", False))
        type_name = spec.get("type")

        if type_name not in _TYPE_MAP:
            raise ValueError(f"unsupported type: {type_name}")

        if field not in record:
            if not nullable:
                errors.append(f"missing required field: {field}")
            continue

        value = record[field]
        if value is None:
            if not nullable:
                errors.append(f"null not allowed: {field}")
            continue

        expected = _TYPE_MAP[type_name]

        # bool is a subclass of int in Python; do not accept it as a
        # numeric field unless the schema explicitly says bool.
        if type_name in {"int", "float"} and isinstance(value, bool):
            errors.append(f"wrong type for {field}: bool")
            continue

        if not isinstance(value, expected):
            errors.append(
                f"wrong type for {field}: {type(value).__name__}"
            )

    return errors


def schema_compatibility(
    old_schema: dict[str, dict[str, Any]],
    new_schema: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Simple backward-read compatibility policy.

    Existing fields cannot disappear/change type.
    New fields must be nullable.
    """
    issues: list[str] = []

    for field, old_spec in old_schema.items():
        if field not in new_schema:
            issues.append(f"removed field: {field}")
            continue

        new_spec = new_schema[field]
        if old_spec.get("type") != new_spec.get("type"):
            issues.append(f"type changed: {field}")

        if (
            old_spec.get("nullable", False)
            and not new_spec.get("nullable", False)
        ):
            issues.append(f"nullable became required: {field}")

    for field, new_spec in new_schema.items():
        if field not in old_schema and not new_spec.get(
            "nullable",
            False,
        ):
            issues.append(f"new required field: {field}")

    return len(issues) == 0, issues


def deduplicate_latest(
    records: list[dict[str, Any]],
    *,
    key_field: str,
    timestamp_field: str,
) -> list[dict[str, Any]]:
    latest: dict[Any, dict[str, Any]] = {}

    for record in records:
        if key_field not in record or timestamp_field not in record:
            raise ValueError("record missing key/timestamp")

        key = record[key_field]
        timestamp = record[timestamp_field]

        current = latest.get(key)
        if (
            current is None
            or timestamp > current[timestamp_field]
        ):
            latest[key] = dict(record)

    return [
        latest[key]
        for key in sorted(latest, key=lambda value: str(value))
    ]


def partition_path(
    record: dict[str, Any],
    *,
    fields: list[str],
) -> str:
    if not fields:
        raise ValueError("fields must be non-empty")

    parts = []
    for field in fields:
        if field not in record:
            raise ValueError(f"missing partition field: {field}")
        value = str(record[field])
        if "/" in value or "=" in value:
            raise ValueError("unsafe partition value")
        parts.append(f"{field}={value}")

    return "/".join(parts)


def watermark_from_max_event_time(
    max_event_time: datetime,
    *,
    allowed_lateness: timedelta,
) -> datetime:
    if allowed_lateness < timedelta(0):
        raise ValueError("allowed_lateness must be non-negative")
    if max_event_time.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")

    return max_event_time - allowed_lateness


def classify_late_event(
    event_time: datetime,
    *,
    watermark: datetime,
) -> str:
    if event_time.tzinfo is None or watermark.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return "late" if event_time < watermark else "on_time"


def idempotency_key(
    *,
    source_event_id: str,
    transform_version: str,
) -> str:
    if not source_event_id or not transform_version:
        raise ValueError("identifiers must be non-empty")

    payload = f"{source_event_id}\0{transform_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lineage_fingerprint(
    *,
    input_versions: dict[str, str],
    transform_version: str,
    output_schema: dict[str, Any],
) -> str:
    payload = {
        "input_versions": input_versions,
        "transform_version": transform_version,
        "output_schema": output_schema,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def small_file_statistics(
    file_sizes_bytes: list[int],
    *,
    target_file_size_bytes: int,
) -> dict[str, float | int]:
    if not file_sizes_bytes:
        raise ValueError("file sizes must be non-empty")
    if any(size < 0 for size in file_sizes_bytes):
        raise ValueError("file sizes must be non-negative")
    if target_file_size_bytes <= 0:
        raise ValueError("target size must be positive")

    total = sum(file_sizes_bytes)
    small = sum(
        size < target_file_size_bytes / 4
        for size in file_sizes_bytes
    )

    return {
        "files": len(file_sizes_bytes),
        "total_bytes": total,
        "average_bytes": total / len(file_sizes_bytes),
        "small_file_fraction": small / len(file_sizes_bytes),
    }


def utc_datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
) -> datetime:
    """Test/example helper with explicit UTC semantics."""
    return datetime(
        year,
        month,
        day,
        hour,
        minute,
        tzinfo=timezone.utc,
    )
