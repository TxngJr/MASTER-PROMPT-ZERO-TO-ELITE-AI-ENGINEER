"""Batch 18 integration: distributed planning, CUDA inspection and AMP."""

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


def run_distributed_plan() -> dict[str, Any]:
    distributed = _load_module(
        "52-distributed-training/src/distributed_numpy.py",
        "batch18_distributed_math",
    )

    world_size = 4
    sample_count = 103
    shards = [
        distributed.shard_indices(
            sample_count,
            world_size=world_size,
            rank=rank,
        )
        for rank in range(world_size)
    ]

    local_gradients = [
        np.array([rank + 1.0, 2.0 * (rank + 1.0)])
        for rank in range(world_size)
    ]
    reduced = distributed.all_reduce_mean(local_gradients)

    parameter_bytes = 2_000_000_000
    gradient_bytes = 2_000_000_000
    optimizer_bytes = 4_000_000_000

    memory = {
        f"stage_{stage}": distributed.zero_style_state_bytes(
            parameter_bytes=parameter_bytes,
            gradient_bytes=gradient_bytes,
            optimizer_bytes=optimizer_bytes,
            world_size=world_size,
            stage=stage,
        )
        for stage in range(4)
    }

    ring_payload = distributed.ring_allreduce_per_rank_bytes(
        gradient_bytes,
        world_size=world_size,
    )

    return {
        "mode": "distributed",
        "world_size": world_size,
        "sample_count": sample_count,
        "shard_sizes": [int(len(shard)) for shard in shards],
        "unique_sample_ids": int(
            len(set(np.concatenate(shards).tolist()))
        ),
        "all_reduce_mean": reduced.tolist(),
        "effective_global_batch": distributed.effective_global_batch(
            local_batch=8,
            world_size=world_size,
            accumulation_steps=4,
        ),
        "ring_allreduce_bytes_per_rank": ring_payload,
        "zero_style_state_bytes": memory,
        "pipeline_efficiency": distributed.pipeline_efficiency(
            stages=4,
            microbatches=16,
        ),
    }


def run_gpu_inspection() -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "mode": "gpu",
        "cuda_available": available,
        "torch_version": torch.__version__,
    }

    if not available:
        report["device_count"] = 0
        report["message"] = "CPU-only runtime: CUDA-specific execution skipped."
        return report

    device_count = torch.cuda.device_count()
    report["device_count"] = int(device_count)

    devices = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": int(properties.total_memory),
                "multiprocessor_count": int(
                    properties.multi_processor_count
                ),
                "major": int(properties.major),
                "minor": int(properties.minor),
            }
        )

    report["devices"] = devices

    # Small synchronized CUDA smoke.
    x = torch.randn(256, 256, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    report["smoke_shape"] = list(y.shape)
    report["memory_allocated"] = int(
        torch.cuda.memory_allocated()
    )

    return report


def run_mixed_precision(
    *,
    steps: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device_type = "cuda" if use_cuda else "cpu"
    device = torch.device(device_type)
    amp_dtype = torch.float16 if use_cuda else torch.bfloat16

    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.GELU(),
        nn.Linear(32, 4),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    features = torch.randn(32, 16, device=device)
    targets = torch.randint(0, 4, (32,), device=device)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_cuda,
    )

    losses: list[float] = []
    scales: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device_type,
            dtype=amp_dtype,
        ):
            logits = model(features)
            loss = F.cross_entropy(logits, targets)

        if use_cuda:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            scales.append(float(scaler.get_scale()))
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()
            scales.append(1.0)

        losses.append(float(loss.detach().float().cpu()))

    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=device_type,
            dtype=amp_dtype,
        ):
            output = model(features[:2])

    return {
        "mode": "precision",
        "device": str(device),
        "autocast_dtype": str(amp_dtype),
        "uses_grad_scaler": use_cuda,
        "steps": steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "scales": scales,
        "output_finite": bool(
            torch.isfinite(output.float()).all().item()
        ),
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 18 systems/mixed-precision lab."
    )
    parser.add_argument(
        "--mode",
        choices=["distributed", "gpu", "precision"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch18"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "distributed":
        report = run_distributed_plan()
    elif args.mode == "gpu":
        report = run_gpu_inspection()
    else:
        report = run_mixed_precision(
            steps=args.steps,
            seed=args.seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.mode}.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"mode: {report['mode']}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
