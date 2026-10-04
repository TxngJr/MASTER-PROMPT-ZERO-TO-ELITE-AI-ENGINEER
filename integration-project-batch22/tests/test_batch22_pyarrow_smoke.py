from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_write_read_parquet_and_row_groups(tmp_path: Path) -> None:
    table = pa.table(
        {
            "event_id": ["e1", "e2", "e3", "e4"],
            "score": [0.1, 0.2, 0.3, 0.4],
        }
    )

    path = tmp_path / "events.parquet"
    pq.write_table(
        table,
        path,
        row_group_size=2,
    )

    parquet_file = pq.ParquetFile(path)

    assert parquet_file.metadata.num_rows == 4
    assert parquet_file.metadata.num_row_groups == 2

    projected = pq.read_table(
        path,
        columns=["event_id"],
    )
    assert projected.column_names == ["event_id"]
    assert projected.num_rows == 4
