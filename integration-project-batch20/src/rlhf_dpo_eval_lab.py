"""Batch 20 integration: reward modeling, DPO, and reproducible evaluation."""

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


def make_preference_problem(
    *,
    num_prompts: int = 8,
    num_responses: int = 4,
    feature_dim: int = 6,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    if min(num_prompts, num_responses, feature_dim) <= 0:
        raise ValueError("dimensions must be positive")

    rng = np.random.default_rng(seed)

    response_features = rng.normal(
        size=(num_prompts, num_responses, feature_dim)
    ).astype(np.float32)

    hidden_reward_weight = rng.normal(
        size=(feature_dim,)
    ).astype(np.float32)

    true_utilities = (
        response_features @ hidden_reward_weight
    ).astype(np.float32)

    chosen = np.argmax(true_utilities, axis=1)
    rejected = np.argmin(true_utilities, axis=1)

    return {
        "response_features": response_features,
        "true_utilities": true_utilities,
        "chosen": chosen.astype(np.int64),
        "rejected": rejected.astype(np.int64),
    }


def run_reward_model(
    *,
    steps: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)
    problem = make_preference_problem(seed=seed)

    features = torch.from_numpy(problem["response_features"])
    chosen = torch.from_numpy(problem["chosen"])
    rejected = torch.from_numpy(problem["rejected"])

    model = nn.Linear(
        features.shape[-1],
        1,
        bias=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-2,
        weight_decay=0.0,
    )

    batch_index = torch.arange(features.shape[0])

    losses: list[float] = []

    for _ in range(steps):
        scores = model(features).squeeze(-1)

        chosen_scores = scores[batch_index, chosen]
        rejected_scores = scores[batch_index, rejected]

        loss = F.softplus(
            -(chosen_scores - rejected_scores)
        ).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach()))

    with torch.inference_mode():
        scores = model(features).squeeze(-1)
        chosen_scores = scores[batch_index, chosen]
        rejected_scores = scores[batch_index, rejected]

        pair_accuracy = float(
            (chosen_scores > rejected_scores)
            .float()
            .mean()
        )

    return {
        "mode": "reward",
        "steps": steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "pair_accuracy": pair_accuracy,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }


def run_dpo(
    *,
    steps: int = 60,
    beta: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    if steps <= 0 or beta <= 0:
        raise ValueError("steps/beta must be positive")

    torch.manual_seed(seed)
    problem = make_preference_problem(seed=seed)

    chosen = torch.from_numpy(problem["chosen"])
    rejected = torch.from_numpy(problem["rejected"])

    num_prompts, num_responses = (
        problem["true_utilities"].shape
    )

    reference_logits = torch.zeros(
        num_prompts,
        num_responses,
    )
    policy_logits = torch.nn.Parameter(
        torch.zeros_like(reference_logits)
    )

    optimizer = torch.optim.AdamW(
        [policy_logits],
        lr=8e-2,
        weight_decay=0.0,
    )

    prompt_index = torch.arange(num_prompts)

    with torch.no_grad():
        reference_log_probs = F.log_softmax(
            reference_logits,
            dim=-1,
        )
        ref_chosen = reference_log_probs[
            prompt_index,
            chosen,
        ]
        ref_rejected = reference_log_probs[
            prompt_index,
            rejected,
        ]

    losses: list[float] = []

    for _ in range(steps):
        policy_log_probs = F.log_softmax(
            policy_logits,
            dim=-1,
        )
        policy_chosen = policy_log_probs[
            prompt_index,
            chosen,
        ]
        policy_rejected = policy_log_probs[
            prompt_index,
            rejected,
        ]

        dpo_logit = beta * (
            (policy_chosen - policy_rejected)
            - (ref_chosen - ref_rejected)
        )

        loss = F.softplus(-dpo_logit).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach()))

    with torch.inference_mode():
        final_log_probs = F.log_softmax(
            policy_logits,
            dim=-1,
        )
        chosen_log_probs = final_log_probs[
            prompt_index,
            chosen,
        ]
        rejected_log_probs = final_log_probs[
            prompt_index,
            rejected,
        ]

        preference_accuracy = float(
            (
                chosen_log_probs
                > rejected_log_probs
            )
            .float()
            .mean()
        )

        predicted = torch.argmax(
            policy_logits,
            dim=-1,
        ).cpu().numpy()

    hidden_best = np.argmax(
        problem["true_utilities"],
        axis=1,
    )

    true_best_accuracy = float(
        np.mean(predicted == hidden_best)
    )

    return {
        "mode": "dpo",
        "steps": steps,
        "beta": beta,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "preference_accuracy": preference_accuracy,
        "true_best_accuracy": true_best_accuracy,
        "trainable_parameters": int(policy_logits.numel()),
    }


def run_evaluation(
    *,
    seed: int = 42,
) -> dict[str, Any]:
    metrics = _load_module(
        "60-llm-evaluation/src/evaluation_metrics.py",
        "batch20_evaluation_metrics",
    )

    baseline = np.array(
        [1, 0, 1, 0, 1, 0, 0, 1],
        dtype=float,
    )
    aligned = np.array(
        [1, 1, 1, 0, 1, 1, 0, 1],
        dtype=float,
    )

    baseline_mean, baseline_low, baseline_high = (
        metrics.bootstrap_mean_interval(
            baseline,
            num_resamples=500,
            seed=seed,
        )
    )

    difference, diff_low, diff_high = (
        metrics.paired_bootstrap_difference(
            aligned,
            baseline,
            num_resamples=500,
            seed=seed,
        )
    )

    contamination = metrics.exact_contamination_rate(
        [
            "held out preference prompt",
            "unique evaluation question",
        ],
        [
            "training instruction",
            "Held out preference prompt!",
        ],
    )

    pairwise = metrics.pairwise_rates(
        ["a", "a", "tie", "a", "b"]
    )

    return {
        "mode": "evaluate",
        "baseline_mean": baseline_mean,
        "baseline_interval": [
            baseline_low,
            baseline_high,
        ],
        "aligned_minus_baseline": difference,
        "difference_interval": [
            diff_low,
            diff_high,
        ],
        "contamination_rate": contamination,
        "pairwise_rates": pairwise,
        "manifest": {
            "seed": seed,
            "num_resamples": 500,
            "paired_comparison": True,
            "normalization": "chapter60-v1",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 20 RLHF/DPO/evaluation lab."
    )
    parser.add_argument(
        "--mode",
        choices=["reward", "dpo", "evaluate"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch20"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "reward":
        report = run_reward_model(
            steps=args.steps,
            seed=args.seed,
        )
    elif args.mode == "dpo":
        report = run_dpo(
            steps=args.steps,
            beta=args.beta,
            seed=args.seed,
        )
    else:
        report = run_evaluation(seed=args.seed)

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
