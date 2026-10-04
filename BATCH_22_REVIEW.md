# Batch 22 Review — Chapters 64–66

## Chapters

- 64 — Deploy AI: API / Docker / Kubernetes / Cloud
- 65 — MLOps
- 66 — AI Data Engineering

## Deploy AI Skills

- immutable deployment artifacts
- Docker layers / runtime images
- API contracts
- Kubernetes Deployment / ReplicaSet / Pod / Service
- Gateway vs legacy Ingress
- ConfigMap / Secret
- startup / readiness / liveness probes
- GPU device resources
- rolling updates
- HPA / custom metrics
- canary / blue-green
- cloud building blocks
- graceful termination
- rollback criteria

## MLOps Skills

- experiment tracking
- run fingerprinting
- artifact hashes
- model registry
- versions / aliases / tags
- promotion gates
- CI / CD / CT
- offline vs online metrics
- data / prediction / concept drift
- PSI
- data-quality monitoring
- training-serving skew
- champion / challenger
- retraining policy
- lineage / rollback

## AI Data Engineering Skills

- ETL / ELT
- lake / warehouse / lakehouse concepts
- Apache Parquet
- row groups / column chunks
- projection / predicate pruning
- partitioning
- small-file compaction
- batch / streaming
- event time / processing time
- watermarks / late data
- idempotency
- exactly-once boundaries
- data contracts
- schema evolution
- backfills
- orchestration
- lineage
- point-in-time correctness

## Implemented From Scratch

### Chapter 64
- desired_replicas_from_metric
- rolling_update_bounds
- canary_request_counts
- availability
- capacity_with_headroom
- gpu_pool_capacity

### Chapter 65
- canonical_json
- experiment_fingerprint
- artifact_sha256
- population_stability_index
- missing_rate
- promotion_gate
- registry_alias_update
- rolling_error_rate

### Chapter 66
- validate_record
- schema_compatibility
- deduplicate_latest
- partition_path
- watermark_from_max_event_time
- classify_late_event
- idempotency_key
- lineage_fingerprint
- small_file_statistics

## Integration Project

The Batch 22 integration connects:

1. data contract validation / dedup / partition / lineage
2. experiment fingerprint / drift / promotion gate / registry alias
3. HPA-style scaling / rolling-update / GPU capacity / canary plan
4. one final rollout gate requiring data + model + infrastructure readiness

## Current Ecosystem Audit

### Kubernetes

Current official Kubernetes documentation:
- uses autoscaling/v2 for stable HPA
- supports custom/external metrics
- manages GPUs through vendor device plugins / extended resources
- recommends Gateway API over Ingress for new feature development because Ingress is frozen

### MLflow

Current MLflow Tracking records parameters, code versions, metrics and artifacts.

Current MLflow Model Registry provides:
- model versions
- aliases
- tags
- metadata
- run/model lineage

### Apache Parquet

Current Parquet documentation describes:
- column-oriented storage
- row groups
- column chunks
- file metadata
- compatibility rules for evolving format features

Batch 22 includes an optional PyArrow smoke workflow that writes a real Parquet file, creates two row groups and performs column projection.

## Methodology Audit

### Deployment
- model quality is separate from Pod readiness
- rollout surge is checked against scarce GPU capacity
- immutable artifact IDs are required for rollback
- HPA signals are workload-specific rather than CPU-only by assumption

### MLOps
- run identity includes code + data + config
- aliases are movable pointers, not historical identities
- drift is not treated as automatic proof of quality failure
- promotion gates combine quality + systems requirements

### Data
- schema contract runs before downstream promotion
- latest-version dedup is deterministic
- partition values are validated
- retries can use deterministic idempotency keys
- lineage hashes include source versions + transform + schema
- event-time late-data behavior is explicit

## Security / Governance Audit

- real credentials are never stored in example manifests
- Kubernetes Secret base64 is not described as encryption
- least-privilege / external-secret-manager concepts are documented
- raw production data and prompt contents should follow access/privacy policies
- lineage records identifiers and versions rather than copying sensitive payloads unnecessarily

## Interpretation Audit

- a green Deployment does not prove a good model
- drift does not automatically require retraining
- a registry alias is not an immutable version
- pipeline success does not prove data correctness
- physical schema compatibility does not prove semantic compatibility
- exactly-once guarantees must name their system boundary
- Parquet file-count and partition choices materially affect performance

## Exit Gate

Before Chapter 67:

1. Batch 22 Core CI passes
2. Batch 22 PyArrow smoke passes
3. build immutable Docker/Kubernetes deployment identity
4. derive HPA-style replica math
5. explain GPU scheduling/device plugins
6. build experiment + artifact fingerprints
7. use explicit promotion gates
8. compute a drift metric and explain its limits
9. validate schema contracts
10. handle event-time watermarks / late data
11. demonstrate idempotent dedup
12. produce end-to-end lineage
13. pass the full lifecycle rollout gate
