"""Batch 23 integration: reliable distributed AI + guardrails + XAI."""

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


def run_distributed() -> dict[str, Any]:
    distributed = _load_module(
        "67-ai-distributed-systems/src/distributed_systems.py",
        "batch23_distributed",
    )

    utilization = distributed.queue_utilization(
        arrival_rate=18.0,
        service_rate_per_worker=5.0,
        workers=4,
    )
    admitted, rejected = distributed.bounded_admission(
        queue_size=8,
        queue_capacity=10,
        incoming=5,
    )
    placements = {
        request: distributed.rendezvous_nodes(
            request,
            ["worker-a", "worker-b", "worker-c"],
            replicas=2,
        )
        for request in ["req-1", "req-2", "req-3"]
    }

    state = distributed.CircuitBreakerState()
    state = distributed.circuit_breaker_record(
        state,
        success=False,
        failure_threshold=2,
    )
    state = distributed.circuit_breaker_record(
        state,
        success=False,
        failure_threshold=2,
    )

    return {
        "mode": "distributed",
        "queue_utilization": utilization,
        "admitted": admitted,
        "rejected": rejected,
        "placements": placements,
        "retry_delays": [
            distributed.exponential_backoff(
                attempt,
                base_seconds=0.25,
                cap_seconds=2.0,
            )
            for attempt in range(5)
        ],
        "circuit_state": state.state,
    }


def run_guardrails() -> dict[str, Any]:
    guardrails = _load_module(
        "68-ai-safety-alignment-guardrails/src/guardrails.py",
        "batch23_guardrails",
    )

    documents = [
        {
            "id": "course-public",
            "allowed_groups": ["students", "staff"],
        },
        {
            "id": "staff-only",
            "allowed_groups": ["staff"],
        },
    ]
    visible = guardrails.retrieval_acl_filter(
        documents,
        principal_groups={"students"},
    )

    read_decision = guardrails.permission_decision(
        tool="read_progress",
        allowlisted_tools={"read_progress", "update_progress"},
        write_tools={"update_progress"},
        user_approved=False,
    )
    write_without_approval = guardrails.permission_decision(
        tool="update_progress",
        allowlisted_tools={"read_progress", "update_progress"},
        write_tools={"update_progress"},
        user_approved=False,
    )
    write_with_approval = guardrails.permission_decision(
        tool="update_progress",
        allowlisted_tools={"read_progress", "update_progress"},
        write_tools={"update_progress"},
        user_approved=True,
    )

    metrics = guardrails.safety_metrics(
        [True, True, False, False, False],
        [True, True, False, False, False],
    )

    return {
        "mode": "guardrails",
        "visible_document_ids": [
            record["id"] for record in visible
        ],
        "read_decision": read_decision,
        "write_without_approval": write_without_approval,
        "write_with_approval": write_with_approval,
        "budget_allows": guardrails.budget_allows(
            used=70,
            requested=20,
            limit=100,
        ),
        "safety_metrics": metrics,
    }


def run_xai(
    *,
    seed: int = 42,
) -> dict[str, Any]:
    xai = _load_module(
        "69-interpretability-xai/src/xai.py",
        "batch23_xai",
    )

    weights = np.array([2.0, -1.0, 0.5])

    def function(values):
        return float(np.dot(weights, values) + 3.0)

    point = np.array([1.5, 2.0, -4.0])
    baseline = np.zeros_like(point)

    attrs = xai.integrated_gradients(
        function,
        point,
        baseline=baseline,
        steps=40,
    )
    gap = xai.completeness_gap(
        function,
        point,
        attrs,
        baseline=baseline,
    )

    contributions = weights * point

    def value(subset):
        return float(
            sum(
                contributions[index]
                for index in subset
            )
        )

    shapley = xai.exact_shapley_values(
        value,
        num_features=3,
    )

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(250, 3))
    targets = 4.0 * features[:, 0] - features[:, 1]

    def predict(values):
        return 4.0 * values[:, 0] - values[:, 1]

    def negative_mse(y, prediction):
        return -float(np.mean((y - prediction) ** 2))

    permutation_mean, permutation_std = (
        xai.permutation_importance(
            predict,
            features,
            targets,
            metric=negative_mse,
            repeats=5,
            seed=seed,
        )
    )

    return {
        "mode": "xai",
        "integrated_gradients": attrs.tolist(),
        "completeness_gap": float(gap),
        "exact_shapley": shapley.tolist(),
        "ig_vs_shapley_similarity": (
            xai.cosine_attribution_similarity(
                attrs,
                shapley,
            )
        ),
        "permutation_mean": permutation_mean.tolist(),
        "permutation_std": permutation_std.tolist(),
    }


def run_full() -> dict[str, Any]:
    distributed = run_distributed()
    guardrails = run_guardrails()
    xai = run_xai()

    reliable = (
        distributed["queue_utilization"] < 1.0
        and distributed["circuit_state"] == "open"
    )
    safety_ok = (
        guardrails["visible_document_ids"] == ["course-public"]
        and guardrails["read_decision"] == "allow"
        and guardrails["write_without_approval"]
        == "approval_required"
        and guardrails["write_with_approval"] == "allow"
        and guardrails["budget_allows"]
    )
    explanation_ok = (
        abs(xai["completeness_gap"]) < 1e-6
        and xai["ig_vs_shapley_similarity"] > 0.999
    )

    return {
        "mode": "full",
        "system_ready": (
            reliable
            and safety_ok
            and explanation_ok
        ),
        "distributed": distributed,
        "guardrails": guardrails,
        "xai": xai,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 23 reliable/safe/interpretable AI lab."
    )
    parser.add_argument(
        "--mode",
        choices=["distributed", "guardrails", "xai", "full"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch23"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "distributed":
        report = run_distributed()
    elif args.mode == "guardrails":
        report = run_guardrails()
    elif args.mode == "xai":
        report = run_xai()
    else:
        report = run_full()

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
