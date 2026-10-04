# Chapter 66 — AI Data Engineering

## 1. Why AI Needs Data Engineering

A training script cannot compensate for unreliable data.

~~~text
sources
↓
ingestion
↓
validation / contracts
↓
transform
↓
columnar storage / partitions
↓
training / features / analytics
↓
lineage + monitoring
~~~

## 2. Learning Objectives

- distinguish ETL and ELT
- understand data lakes, warehouses and lakehouse concepts
- understand Parquet's columnar layout
- reason about row groups and column chunks
- design useful partitions
- avoid the small-file problem
- explain projection/predicate pruning
- build batch and streaming pipelines
- distinguish event time and processing time
- reason about watermarks and late data
- build idempotent transformations
- understand exactly-once claims carefully
- design data contracts
- reason about schema evolution
- implement deduplication/backfills
- preserve lineage
- prevent training-serving skew

## 3. ETL vs ELT

### ETL
~~~text
Extract → Transform → Load
~~~

### ELT
~~~text
Extract → Load → Transform
~~~

Modern data platforms often combine both patterns depending on cost, governance and latency.

## 4. Data Lake

Object storage containing raw and processed datasets.

Strengths:
- cheap scalable storage
- flexible formats
- training-data snapshots

Risks:
- weak governance
- inconsistent schemas
- duplicated/unknown data

## 5. Warehouse

Optimized for structured analytics, governed schemas and SQL workloads.

AI systems often move data between operational stores, lakes and warehouses rather than choosing only one.

## 6. Lakehouse Concept

A lakehouse adds table-management features such as transactions, metadata and schema management on object-storage-style data.

Specific implementations vary; understand the abstraction first.

## 7. Why Parquet

Apache Parquet is a column-oriented file format designed for efficient storage and retrieval with compression and encodings.

Column orientation helps analytical/training scans that read a subset of columns.

## 8. Parquet Physical Shape

Conceptually:

~~~text
File
├── Row Group 0
│   ├── Column Chunk A
│   ├── Column Chunk B
│   └── Column Chunk C
├── Row Group 1
│   ├── Column Chunk A
│   ├── Column Chunk B
│   └── Column Chunk C
└── File Metadata
~~~

Readers use metadata to locate needed column chunks.

## 9. Projection Pushdown

If a dataset has 100 columns but a job needs 5, columnar readers can avoid reading irrelevant columns when the engine/file layout supports it.

Do not estimate savings as exactly 95% without considering compression and metadata.

## 10. Predicate / Row-Group Pruning

Statistics/metadata may allow engines to skip row groups that cannot satisfy a filter.

Example:

~~~text
filter: event_date = 2026-10-04
~~~

Partitioning and row-group statistics can reduce scanned data dramatically.

## 11. Partitioning

Partition by fields that are frequently used for pruning and have manageable cardinality.

Typical:
- date
- region
- dataset version

Bad partition key:
- user_id with millions of tiny partitions

## 12. Small-File Problem

Thousands/millions of tiny files create:
- metadata overhead
- object-store requests
- scheduler overhead
- inefficient reads

Compaction rewrites small files into fewer larger files.

## 13. Batch Pipelines

Process bounded datasets at intervals:
- hourly
- daily
- backfill
- snapshot rebuild

Benefits:
- simpler reasoning
- reproducible snapshots

## 14. Streaming Pipelines

Process an unbounded event stream continuously or in micro-batches.

Challenges:
- ordering
- duplicates
- late data
- state
- retries
- exactly-once semantics

## 15. Event Time vs Processing Time

Event time:
~~~text
when event happened
~~~

Processing time:
~~~text
when pipeline processed it
~~~

Network delay means they can differ substantially.

## 16. Watermarks

A watermark summarizes how far event time is believed to have progressed.

Simple educational model:

~~~text
watermark = max_event_time_seen - allowed_lateness
~~~

Events older than the watermark require an explicit late-data policy.

## 17. Late Data

Policies:
- update previous window
- send to late-event stream
- trigger correction/backfill
- drop only when business rules permit

Silent dropping is dangerous.

## 18. Idempotency

A retry should not create duplicate logical output.

Use deterministic identities such as:

~~~text
hash(source_event_id + transform_version)
~~~

or transactional upsert keys.

## 19. Exactly Once

'Exactly once' depends on system boundaries.

A streaming engine may guarantee exactly-once state updates while an external API call still happens twice after retry.

Define exactly-once for:
- source offsets
- state
- sink writes
- external side effects

## 20. Deduplication

Possible identities:
- event_id
- business key + event timestamp
- content hash

Choose whether to keep:
- first occurrence
- latest version
- highest-quality version

## 21. Data Contract

A contract specifies:
- fields
- types
- required/nullable
- semantics
- valid ranges/categories
- freshness/volume expectations

Contracts catch breaking producer changes before corrupted training data spreads.

## 22. Schema Evolution

Changes can be:
- additive nullable field
- renamed field
- type change
- semantic change

Technical readability does not mean semantic compatibility.

Apache Parquet's format evolves through features with different reader/writer compatibility; current official docs explicitly distinguish forward-compatible and forward-incompatible additions.

## 23. Backfills

A backfill recomputes historical ranges.

Requirements:
- deterministic transform version
- bounded date/range
- isolated output before promotion
- idempotent reruns
- validation against current production data

## 24. Orchestration

An orchestrator manages:
- dependencies
- retries
- schedules
- backfills
- SLAs
- task metadata

DAG success still requires checking data-quality gates.

## 25. Lineage

Lineage records:

~~~text
source tables/files/events
↓
transform version
↓
output dataset
↓
training run
↓
model
~~~

This connects Chapter 66 directly to MLOps.

## 26. Training Data Snapshot

Training should reference an immutable or reconstructable snapshot.

Do not train from a mutable query whose result silently changes later.

## 27. Feature Pipelines

Features should have:
- definition
- source
- timestamp semantics
- version
- freshness
- offline/online consistency tests

## 28. Point-in-Time Correctness

For supervised learning, a feature must only use information available at prediction time.

Otherwise future information leaks into training.

## 29. Data Quality

Monitor:
- schema validity
- null rate
- uniqueness
- range
- freshness
- volume
- duplicate rate
- partition completeness

## 30. From Scratch

src/data_engineering.py implements:

- validate_record
- schema_compatibility
- deduplicate_latest
- partition_path
- watermark_from_max_event_time
- classify_late_event
- idempotency_key
- lineage_fingerprint
- small_file_statistics

## 31. Optional Parquet Lab

Install:

~~~bash
python -m pip install -r requirements-batch22-data.txt
~~~

Then write/read a partitioned PyArrow Parquet dataset and inspect file/row-group metadata.

## 32. Common Mistakes

1. user ID used as high-cardinality partition
2. millions of tiny Parquet files
3. event time confused with ingestion time
4. late events silently discarded
5. retries create duplicate output
6. exactly-once assumed across external side effects
7. nullable field addition assumed semantically harmless
8. mutable query used as training-data version
9. future features leak into historical labels
10. pipeline success considered equivalent to data correctness

## 33. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 34. Checklist

- [ ] ETL / ELT
- [ ] lake / warehouse / lakehouse
- [ ] Parquet
- [ ] partitioning / compaction
- [ ] batch / stream
- [ ] event time / watermarks
- [ ] idempotency
- [ ] contracts
- [ ] schema evolution
- [ ] backfills
- [ ] lineage
- [ ] point-in-time correctness

## 35. What's Next

Batch 23 combines systems reliability, safety and interpretability: distributed AI systems, guardrails/alignment and XAI.