# Chapter 66 Exercises

1. Define a schema/data contract for an event dataset.
2. Test additive nullable vs breaking schema changes.
3. Deduplicate latest event versions.
4. Design date/region partitions and explain cardinality.
5. Detect a small-file problem and plan compaction.
6. Compute event-time watermarks and classify late events.
7. Build deterministic idempotency keys.
8. Design a backfill that can be rerun safely.
9. Build a lineage fingerprint across two input datasets.
10. Identify point-in-time leakage in a feature pipeline.

Challenge:
- write a partitioned PyArrow Parquet dataset
- inspect row-group metadata and measure projection/filter scan savings
