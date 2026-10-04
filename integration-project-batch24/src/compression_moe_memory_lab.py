"""Batch 24 integration: compression, sparse MoE, and long-context memory."""

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


def run_compression(
    *,
    seed: int = 42,
) -> dict[str, Any]:
    compression = _load_module(
        "70-model-compression/src/compression.py",
        "batch24_compression",
    )

    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(32, 64))

    pruned, mask = compression.magnitude_prune(
        weights,
        fraction=0.5,
    )

    teacher_logits = rng.normal(
        size=(64, 5),
    )
    student_logits = teacher_logits + rng.normal(
        scale=0.15,
        size=teacher_logits.shape,
    )

    kd = compression.distillation_loss(
        teacher_logits,
        student_logits,
        temperature=2.0,
    )

    return {
        "mode": "compression",
        "original_nonzeros": compression.nonzero_count(weights),
        "pruned_nonzeros": compression.nonzero_count(pruned),
        "sparsity": compression.sparsity(pruned),
        "masked_values": int(np.count_nonzero(~mask)),
        "distillation_loss": kd,
        "ideal_dense_fp16_bytes": int(weights.size * 2),
    }


def run_moe() -> dict[str, Any]:
    moe = _load_module(
        "71-mixture-of-experts/src/moe.py",
        "batch24_moe",
    )

    router_logits = np.array(
        [
            [5.0, 4.0, 1.0, 0.0],
            [4.5, 4.0, 0.0, 1.0],
            [0.0, 1.0, 5.0, 4.0],
            [1.0, 0.0, 4.5, 4.0],
            [5.0, 0.0, 4.0, 1.0],
            [0.0, 5.0, 1.0, 4.0],
            [4.0, 1.0, 0.0, 5.0],
            [1.0, 4.0, 5.0, 0.0],
        ]
    )

    indices, weights, probabilities = moe.top_k_router(
        router_logits,
        top_k=2,
    )

    capacity = moe.expert_capacity(
        tokens=len(router_logits),
        num_experts=4,
        top_k=2,
        capacity_factor=1.25,
    )
    accepted = moe.apply_capacity(
        indices,
        weights,
        num_experts=4,
        capacity=capacity,
    )

    top1 = indices[:, 0]
    loads = moe.expert_load(
        indices,
        num_experts=4,
    )

    return {
        "mode": "moe",
        "capacity": capacity,
        "expert_load": loads.tolist(),
        "dropped_assignment_fraction": (
            moe.dropped_assignment_fraction(accepted)
        ),
        "load_balance_loss": moe.switch_load_balance_loss(
            probabilities,
            top1,
        ),
        "active_parameter_fraction": moe.active_parameter_fraction(
            shared_parameters=20_000_000,
            expert_parameters_each=10_000_000,
            num_experts=8,
            active_experts=2,
        ),
    }


def run_context() -> dict[str, Any]:
    context = _load_module(
        "72-long-context-memory/src/long_context.py",
        "batch24_context",
    )

    sequence_length = 8192
    window = 512

    dense_pairs = context.causal_dense_attention_pairs(
        sequence_length
    )
    sliding_pairs = context.sliding_window_attention_pairs(
        sequence_length,
        window=window,
    )

    selected = context.select_context_segments(
        [1200, 800, 400, 300],
        [0.95, 0.80, 0.70, 0.60],
        token_budget=2000,
    )

    ranked = ["memory-a", "memory-c", "memory-b", "memory-d"]
    relevant = {"memory-a", "memory-b"}

    return {
        "mode": "context",
        "dense_attention_pairs": dense_pairs,
        "sliding_attention_pairs": sliding_pairs,
        "pair_reduction": context.attention_pair_reduction(
            sequence_length,
            window=window,
        ),
        "full_kv_bytes": context.kv_cache_bytes(
            layers=24,
            sequence_length=sequence_length,
            kv_heads=8,
            head_dim=128,
            bytes_per_element=2,
        ),
        "sliding_cache_tokens": (
            context.bounded_sliding_cache_tokens(
                sequence_length,
                window=window,
            )
        ),
        "selected_segments": selected,
        "memory_recall_at_3": context.retrieval_recall_at_k(
            ranked,
            relevant,
            k=3,
        ),
    }


def run_full() -> dict[str, Any]:
    compression = run_compression()
    moe = run_moe()
    context = run_context()

    compression_ok = (
        compression["sparsity"] >= 0.49
        and np.isfinite(compression["distillation_loss"])
    )
    moe_ok = (
        moe["dropped_assignment_fraction"] <= 0.25
        and 0.0 < moe["active_parameter_fraction"] < 1.0
    )
    context_ok = (
        context["pair_reduction"] > 2.0
        and context["memory_recall_at_3"] == 1.0
    )

    return {
        "mode": "full",
        "system_ready": (
            compression_ok
            and moe_ok
            and context_ok
        ),
        "compression": compression,
        "moe": moe,
        "context": context,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 24 compression/MoE/context lab."
    )
    parser.add_argument(
        "--mode",
        choices=["compression", "moe", "context", "full"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch24"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "compression":
        report = run_compression()
    elif args.mode == "moe":
        report = run_moe()
    elif args.mode == "context":
        report = run_context()
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
