from pathlib import Path
import importlib.util
from datetime import timedelta


MODULE_PATH = Path(__file__).parents[1] / "src" / "data_engineering.py"
SPEC = importlib.util.spec_from_file_location("data_engineering_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_validate_record() -> None:
    schema = {
        "id": {"type": "int", "nullable": False},
        "text": {"type": "str", "nullable": False},
        "score": {"type": "float", "nullable": True},
    }
    assert mod.validate_record(
        {"id": 1, "text": "hello", "score": None},
        schema,
    ) == []

    errors = mod.validate_record(
        {"id": "1", "text": "hello"},
        schema,
    )
    assert any("wrong type" in error for error in errors)


def test_schema_additive_nullable_is_compatible() -> None:
    old = {
        "id": {"type": "int", "nullable": False},
    }
    new = {
        "id": {"type": "int", "nullable": False},
        "note": {"type": "str", "nullable": True},
    }
    compatible, issues = mod.schema_compatibility(old, new)
    assert compatible
    assert issues == []


def test_deduplicate_latest() -> None:
    records = [
        {"id": "a", "ts": 1, "value": 10},
        {"id": "a", "ts": 2, "value": 20},
        {"id": "b", "ts": 1, "value": 30},
    ]
    output = mod.deduplicate_latest(
        records,
        key_field="id",
        timestamp_field="ts",
    )
    assert output == [
        {"id": "a", "ts": 2, "value": 20},
        {"id": "b", "ts": 1, "value": 30},
    ]


def test_partition_path() -> None:
    path = mod.partition_path(
        {"date": "2026-10-04", "region": "th"},
        fields=["date", "region"],
    )
    assert path == "date=2026-10-04/region=th"


def test_watermark_lateness() -> None:
    maximum = mod.utc_datetime(2026, 10, 4, 12, 0)
    watermark = mod.watermark_from_max_event_time(
        maximum,
        allowed_lateness=timedelta(minutes=10),
    )

    late = mod.utc_datetime(2026, 10, 4, 11, 49)
    on_time = mod.utc_datetime(2026, 10, 4, 11, 55)

    assert mod.classify_late_event(
        late,
        watermark=watermark,
    ) == "late"
    assert mod.classify_late_event(
        on_time,
        watermark=watermark,
    ) == "on_time"


def test_idempotency_key_is_versioned() -> None:
    first = mod.idempotency_key(
        source_event_id="event-1",
        transform_version="v1",
    )
    second = mod.idempotency_key(
        source_event_id="event-1",
        transform_version="v2",
    )
    assert first != second


def test_lineage_is_order_independent() -> None:
    first = mod.lineage_fingerprint(
        input_versions={"a": "1", "b": "2"},
        transform_version="code-1",
        output_schema={"x": "int"},
    )
    second = mod.lineage_fingerprint(
        input_versions={"b": "2", "a": "1"},
        transform_version="code-1",
        output_schema={"x": "int"},
    )
    assert first == second


def test_small_file_statistics() -> None:
    stats = mod.small_file_statistics(
        [10, 20, 1000, 1200],
        target_file_size_bytes=1000,
    )
    assert stats["files"] == 4
    assert stats["small_file_fraction"] == 0.5
