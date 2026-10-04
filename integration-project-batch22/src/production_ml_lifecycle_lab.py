"""Batch 22 integration: data pipeline, MLOps promotion, deployment planning."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _load_module(relative_path: str, name: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_data_pipeline() -> dict[str, Any]:
    data = _load_module(
        "66-ai-data-engineering/src/data_engineering.py",
        "batch22_data",
    )

    schema = {
        "event_id": {"type": "str", "nullable": False},
        "version": {"type": "int", "nullable": False},
        "date": {"type": "str", "nullable": False},
        "region": {"type": "str", "nullable": False},
        "score": {"type": "float", "nullable": True},
    }

    records = [
        {
            "event_id": "e1",
            "version": 1,
            "date": "2026-10-04",
            "region": "th",
            "score": 0.4,
        },
        {
            "event_id": "e1",
            "version": 2,
            "date": "2026-10-04",
            "region": "th",
            "score": 0.5,
        },
        {
            "event_id": "e2",
            "version": 1,
            "date": "2026-10-04",
            "region": "sg",
            "score": None,
        },
    ]

    validation_errors = [
        error
        for record in records
        for error in data.validate_record(record, schema)
    ]

    deduplicated = data.deduplicate_latest(
        records,
        key_field="event_id",
        timestamp_field="version",
    )

    partitions = [
        data.partition_path(
            record,
            fields=["date", "region"],
        )
        for record in deduplicated
    ]

    lineage = data.lineage_fingerprint(
        input_versions={
            "events": "snapshot-2026-10-04",
            "labels": "labels-v3",
        },
        transform_version="batch22-transform-v1",
        output_schema=schema,
    )

    return {
        "mode": "data",
        "input_records": len(records),
        "output_records": len(deduplicated),
        "validation_errors": validation_errors,
        "partitions": sorted(set(partitions)),
        "lineage_fingerprint": lineage,
    }


def run_mlops_promotion() -> dict[str, Any]:
    mlops = _load_module(
        "65-mlops/src/mlops_utils.py",
        "batch22_mlops",
    )

    run_id = mlops.experiment_fingerprint(
        code_revision="commit-batch22",
        dataset_revision="snapshot-2026-10-04",
        config={
            "model": "candidate-v4",
            "seed": 42,
        },
        environment={
            "python": "3.12",
        },
    )

    metrics = {
        "accuracy": 0.92,
        "p99_ms": 420.0,
        "error_rate": 0.004,
    }
    requirements = {
        "accuracy": (">=", 0.90),
        "p99_ms": ("<=", 500.0),
        "error_rate": ("<=", 0.01),
    }

    promoted, failures = mlops.promotion_gate(
        metrics,
        requirements,
    )

    aliases = {
        "champion": "v3",
        "rollback": "v2",
    }
    if promoted:
        aliases = mlops.registry_alias_update(
            aliases,
            alias="champion",
            version="v4",
        )

    psi = mlops.population_stability_index(
        np.array([40, 35, 25]),
        np.array([38, 36, 26]),
    )

    return {
        "mode": "mlops",
        "run_fingerprint": run_id,
        "promoted": promoted,
        "failures": failures,
        "aliases": aliases,
        "psi": psi,
    }


def run_deployment_plan() -> dict[str, Any]:
    deploy = _load_module(
        "64-deploy-ai/src/deployment_math.py",
        "batch22_deploy",
    )

    desired = deploy.desired_replicas_from_metric(
        current_replicas=4,
        current_metric=75.0,
        target_metric=50.0,
        min_replicas=2,
        max_replicas=10,
    )
    rollout = deploy.rolling_update_bounds(
        desired_replicas=desired,
        max_surge=1,
        max_unavailable=0,
    )
    stable, canary = deploy.canary_request_counts(
        10_000,
        canary_fraction=0.05,
    )

    gpu_capacity = deploy.gpu_pool_capacity(
        nodes=4,
        gpus_per_node=2,
        gpus_per_replica=1,
        reserved_gpus=1,
    )

    return {
        "mode": "deploy",
        "desired_replicas": desired,
        "rollout": rollout,
        "stable_requests": stable,
        "canary_requests": canary,
        "gpu_replica_capacity": gpu_capacity,
        "fits_gpu_pool": rollout["max_total_pods"] <= gpu_capacity,
    }


def run_full_lifecycle() -> dict[str, Any]:
    data_report = run_data_pipeline()
    mlops_report = run_mlops_promotion()
    deployment_report = run_deployment_plan()

    ready_for_rollout = (
        len(data_report["validation_errors"]) == 0
        and mlops_report["promoted"]
        and deployment_report["fits_gpu_pool"]
    )

    return {
        "mode": "full",
        "ready_for_rollout": ready_for_rollout,
        "data": data_report,
        "mlops": mlops_report,
        "deployment": deployment_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 22 production ML lifecycle lab."
    )
    parser.add_argument(
        "--mode",
        choices=["data", "mlops", "deploy", "full"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch22"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "data":
        report = run_data_pipeline()
    elif args.mode == "mlops":
        report = run_mlops_promotion()
    elif args.mode == "deploy":
        report = run_deployment_plan()
    else:
        report = run_full_lifecycle()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.mode}.json"
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"mode: {report['mode']}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
