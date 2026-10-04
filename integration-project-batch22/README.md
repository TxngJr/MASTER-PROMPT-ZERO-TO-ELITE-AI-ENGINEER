# Batch 22 Integration Project — Production ML Lifecycle Lab

Batch 22 connects data engineering, MLOps governance and deployment planning.

## Part A — Data Pipeline

~~~text
raw events
↓
data contract validation
↓
deduplicate latest version
↓
partition planning
↓
lineage fingerprint
~~~

Outputs:
- validation errors
- input/output record counts
- partitions
- lineage fingerprint

## Part B — MLOps Promotion

~~~text
code + data + config
↓
run fingerprint
↓
candidate metrics
↓
promotion gates
↓
registry alias update
~~~

Also computes a drift diagnostic using PSI.

## Part C — Deployment Plan

~~~text
current replicas + load
↓
HPA-style desired replica math
↓
rolling-update surge bound
↓
GPU pool capacity check
↓
5% canary split
~~~

## Part D — Full Lifecycle Gate

A rollout is allowed only when:

~~~text
data contract passes
AND
model promotion gate passes
AND
rollout fits GPU capacity
~~~

This deliberately demonstrates that deployment readiness is more than "the container starts".

## Run

~~~bash
python integration-project-batch22/src/production_ml_lifecycle_lab.py --mode data
python integration-project-batch22/src/production_ml_lifecycle_lab.py --mode mlops
python integration-project-batch22/src/production_ml_lifecycle_lab.py --mode deploy
python integration-project-batch22/src/production_ml_lifecycle_lab.py --mode full
~~~

## Optional Ecosystem Labs

MLflow:

~~~bash
python -m pip install -r requirements-batch22-mlops.txt
~~~

PyArrow:

~~~bash
python -m pip install -r requirements-batch22-data.txt
~~~

## Required Extensions

1. persist a real MLflow run
2. register a model version and alias
3. write a real partitioned Parquet snapshot
4. inspect row-group metadata
5. add delayed-label production metrics
6. add drift/data-quality alert policies
7. generate Kubernetes manifests from immutable artifact IDs
8. add canary quality+latency rollback gates
9. implement backfill lineage
10. rehearse a rollback to a previous champion
