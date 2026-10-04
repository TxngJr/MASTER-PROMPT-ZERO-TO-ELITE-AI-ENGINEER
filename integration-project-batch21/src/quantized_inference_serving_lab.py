"""Batch 21 integration: quantization, inference capacity, serving SLOs."""

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


def run_quantization(
    *,
    seed: int = 42,
) -> dict[str, Any]:
    quant = _load_module(
        "61-quantization/src/quantization_numpy.py",
        "batch21_quant",
    )

    rng = np.random.default_rng(seed)
    weights = rng.normal(
        0.0,
        0.5,
        size=(64, 128),
    )

    results = {}

    for bits in (8, 4):
        q, scale = quant.symmetric_quantize(
            weights,
            bits=bits,
        )
        restored = quant.symmetric_dequantize(
            q,
            scale=scale,
        )

        results[f"int{bits}"] = {
            "mse": quant.quantization_mse(
                weights,
                restored,
            ),
            "ideal_weight_bytes": quant.ideal_storage_bytes(
                weights.size,
                bits_per_value=bits,
            ),
            "compression_vs_fp16": quant.compression_ratio(
                16,
                bits,
            ),
        }

    q_channel, scales = (
        quant.per_channel_symmetric_quantize(
            weights,
            axis=0,
            bits=4,
        )
    )
    restored_channel = np.vstack(
        [
            quant.symmetric_dequantize(
                q_channel[index],
                scale=scales[index],
            )
            for index in range(len(scales))
        ]
    )

    results["int4_per_channel"] = {
        "mse": quant.quantization_mse(
            weights,
            restored_channel,
        ),
        "num_scales": int(len(scales)),
    }

    return {
        "mode": "quantization",
        "shape": list(weights.shape),
        **results,
    }


def run_capacity() -> dict[str, Any]:
    inference = _load_module(
        "62-inference-optimization/src/inference_math.py",
        "batch21_inference_math",
    )

    mha_cache = inference.kv_cache_bytes(
        layers=32,
        batch_size=1,
        sequence_length=4096,
        kv_heads=32,
        head_dim=128,
        bytes_per_element=2,
    )
    gqa_cache = inference.kv_cache_bytes(
        layers=32,
        batch_size=1,
        sequence_length=4096,
        kv_heads=8,
        head_dim=128,
        bytes_per_element=2,
    )

    paging = inference.paged_block_usage(
        4097,
        block_size=16,
    )

    return {
        "mode": "capacity",
        "mha_kv_bytes": mha_cache,
        "gqa_kv_bytes": gqa_cache,
        "gqa_reduction": float(mha_cache / gqa_cache),
        "padding_efficiency": inference.padded_batch_efficiency(
            [256, 512, 1024, 4096]
        ),
        "paged_usage": paging,
        "prefix_saved_tokens": inference.prefix_cache_saved_tokens(
            4096,
            cached_prefix_length=3072,
        ),
        "speculative_acceptance_rate": (
            inference.speculative_acceptance_rate(
                proposed_tokens=100,
                accepted_tokens=72,
            )
        ),
    }


def run_serving() -> dict[str, Any]:
    serving = _load_module(
        "63-llm-serving/src/serving_metrics.py",
        "batch21_serving_metrics",
    )

    latencies = [
        serving.request_latency(
            queue_seconds=queue,
            prefill_seconds=prefill,
            decode_seconds=decode,
            network_seconds=0.02,
        )
        for queue, prefill, decode in [
            (0.02, 0.10, 0.40),
            (0.05, 0.15, 0.50),
            (0.10, 0.20, 0.70),
            (0.25, 0.30, 1.20),
            (0.40, 0.35, 1.60),
        ]
    ]

    stable, canary = serving.canary_split(
        10_000,
        canary_fraction=0.05,
    )

    return {
        "mode": "serving",
        "p50_latency": serving.percentile(latencies, 50),
        "p95_latency": serving.percentile(latencies, 95),
        "p99_latency": serving.percentile(latencies, 99),
        "average_concurrency": serving.little_law_concurrency(
            arrival_rate_per_second=8.0,
            average_latency_seconds=float(np.mean(latencies)),
        ),
        "required_replicas": serving.required_replicas(
            offered_requests_per_second=70.0,
            sustainable_requests_per_second_per_replica=40.0,
            target_utilization=0.7,
        ),
        "stable_requests": stable,
        "canary_requests": canary,
        "success_rate": serving.success_rate(
            successes=9950,
            total_requests=10_000,
        ),
    }


def run_torch_weight_only(
    *,
    bits: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    if bits not in {4, 8}:
        raise ValueError("bits must be 4 or 8")

    quant = _load_module(
        "61-quantization/src/quantization_numpy.py",
        "batch21_quant_torch",
    )

    torch.manual_seed(seed)
    weight = torch.randn(32, 64)
    inputs = torch.randn(16, 64)

    q, scale = quant.symmetric_quantize(
        weight.numpy(),
        bits=bits,
    )
    restored = torch.from_numpy(
        quant.symmetric_dequantize(
            q,
            scale=scale,
        )
    ).to(dtype=weight.dtype)

    original_output = F.linear(inputs, weight)
    quantized_output = F.linear(inputs, restored)

    mse = torch.mean(
        (original_output - quantized_output) ** 2
    )

    return {
        "mode": "torch_weight_only",
        "bits": bits,
        "output_mse": float(mse),
        "outputs_finite": bool(
            torch.isfinite(quantized_output).all()
        ),
        "compression_vs_fp16": float(16 / bits),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 21 quantization/inference/serving lab."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "quantization",
            "capacity",
            "serving",
            "torch-weight-only",
        ],
        required=True,
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch21"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "quantization":
        report = run_quantization(seed=args.seed)
    elif args.mode == "capacity":
        report = run_capacity()
    elif args.mode == "serving":
        report = run_serving()
    else:
        report = run_torch_weight_only(
            bits=args.bits,
            seed=args.seed,
        )

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
