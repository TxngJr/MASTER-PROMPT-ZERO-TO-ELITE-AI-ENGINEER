"""Batch 26 integration: world-model planning, safe simulated robotics, and edge deployment."""

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


def run_world_model() -> dict[str, Any]:
    world = _load_module(
        "76-world-models/src/world_model.py",
        "batch26_world",
    )

    plans = world.enumerate_action_plans(
        [-1.0, 0.0, 1.0],
        horizon=3,
    )

    plan, predicted_return = world.plan_by_model(
        np.array([0.0]),
        plans,
        transition_matrix=np.eye(1),
        action_matrix=np.ones((1, 1)),
        goal=np.array([3.0]),
        gamma=1.0,
        action_penalty=0.01,
    )

    trajectory = world.rollout_linear_dynamics(
        np.array([0.0]),
        plan,
        transition_matrix=np.eye(1),
        action_matrix=np.ones((1, 1)),
    )

    return {
        "mode": "world",
        "plan": plan.tolist(),
        "trajectory": trajectory.tolist(),
        "predicted_return": predicted_return,
        "final_state": float(trajectory[-1, 0]),
    }


def run_robotics() -> dict[str, Any]:
    robotics = _load_module(
        "77-robotics-ai/src/robotics.py",
        "batch26_robotics",
    )

    target = np.array([1.0, -1.0])
    current = np.array([0.0, 0.0])

    raw_action = robotics.proportional_control(
        target,
        current,
        gain=0.8,
    )

    safe_action = robotics.clamp_action(
        raw_action,
        low=np.array([-0.5, -0.5]),
        high=np.array([0.5, 0.5]),
    )

    control_ok = robotics.latency_budget_ok(
        observation_ms=2.0,
        inference_ms=8.0,
        action_dispatch_ms=2.0,
        control_frequency_hz=50.0,
    )

    x, y = robotics.planar_two_link_fk(
        0.0,
        0.0,
        link1=1.0,
        link2=1.0,
    )

    return {
        "mode": "robotics",
        "raw_action": raw_action.tolist(),
        "safe_action": safe_action.tolist(),
        "control_latency_ok": control_ok,
        "fk_position": [x, y],
        "control_period_ms": robotics.control_period_ms(
            frequency_hz=50.0,
        ),
    }


def run_edge() -> dict[str, Any]:
    edge = _load_module(
        "78-edge-ai-tinyml/src/edge_ai.py",
        "batch26_edge",
    )

    fp32_bytes = edge.model_parameter_bytes(
        [(64, 32), (32,), (32, 8), (8,)],
        dtype="float32",
    )
    int8_bytes = edge.model_parameter_bytes(
        [(64, 32), (32,), (32, 8), (8,)],
        dtype="int8",
    )

    peak = edge.peak_memory_bytes(
        model_bytes=int8_bytes,
        peak_activation_bytes=4096,
        arena_bytes=8192,
        runtime_overhead_bytes=4096,
    )

    operator_report = edge.operator_coverage(
        ["MatMul", "Add", "Relu", "MatMul", "Add"],
        {"MatMul", "Add", "Relu"},
    )

    values = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    quantized, scale, zero_point = edge.affine_int8_quantize(
        values
    )
    reconstructed = edge.affine_int8_dequantize(
        quantized,
        scale=scale,
        zero_point=zero_point,
    )

    return {
        "mode": "edge",
        "fp32_parameter_bytes": fp32_bytes,
        "int8_parameter_bytes": int8_bytes,
        "parameter_compression_ratio": fp32_bytes / int8_bytes,
        "peak_required_bytes": peak,
        "fits_64k_budget": edge.fits_memory_budget(
            peak,
            available_bytes=64 * 1024,
            safety_fraction=0.9,
        ),
        "operator_coverage": operator_report,
        "quantization_max_abs_error": float(
            np.max(np.abs(reconstructed - values))
        ),
        "energy_mj": edge.energy_per_inference_mj(
            average_power_watts=1.5,
            latency_ms=8.0,
        ),
    }


def run_full() -> dict[str, Any]:
    world = run_world_model()
    robotics = run_robotics()
    edge = run_edge()

    world_ok = (
        world["final_state"] == 3.0
        and world["plan"] == [[1.0], [1.0], [1.0]]
    )
    robotics_ok = (
        robotics["control_latency_ok"]
        and max(abs(v) for v in robotics["safe_action"]) <= 0.5
    )
    edge_ok = (
        edge["fits_64k_budget"]
        and edge["operator_coverage"]["coverage"] == 1.0
        and edge["parameter_compression_ratio"] == 4.0
    )

    return {
        "mode": "full",
        "system_ready": world_ok and robotics_ok and edge_ok,
        "world": world,
        "robotics": robotics,
        "edge": edge,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 26 world/robotics/edge lab."
    )
    parser.add_argument(
        "--mode",
        choices=["world", "robotics", "edge", "full"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch26"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "world":
        report = run_world_model()
    elif args.mode == "robotics":
        report = run_robotics()
    elif args.mode == "edge":
        report = run_edge()
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
