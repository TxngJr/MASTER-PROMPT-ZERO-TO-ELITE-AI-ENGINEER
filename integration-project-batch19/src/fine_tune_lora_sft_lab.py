"""Batch 19 integration: full fine-tuning, LoRA and instruction SFT."""

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


def make_low_rank_adaptation_problem(
    *,
    samples: int = 128,
    input_dim: int = 12,
    output_dim: int = 6,
    rank: int = 2,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    if min(samples, input_dim, output_dim, rank) <= 0:
        raise ValueError("dimensions must be positive")

    rng = np.random.default_rng(seed)

    x = rng.normal(size=(samples, input_dim)).astype(np.float32)
    base_weight = rng.normal(
        0.0,
        0.35,
        size=(output_dim, input_dim),
    ).astype(np.float32)

    a = rng.normal(
        0.0,
        0.25,
        size=(rank, input_dim),
    ).astype(np.float32)
    b = rng.normal(
        0.0,
        0.25,
        size=(output_dim, rank),
    ).astype(np.float32)

    target_weight = base_weight + b @ a

    return {
        "x": x,
        "base_weight": base_weight,
        "target_weight": target_weight.astype(np.float32),
        "base_targets": (x @ base_weight.T).astype(np.float32),
        "adaptation_targets": (x @ target_weight.T).astype(np.float32),
    }


def run_full_finetune(
    *,
    steps: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)
    problem = make_low_rank_adaptation_problem(seed=seed)

    x = torch.from_numpy(problem["x"])
    base_targets = torch.from_numpy(problem["base_targets"])
    adaptation_targets = torch.from_numpy(
        problem["adaptation_targets"]
    )

    model = nn.Linear(
        x.shape[1],
        adaptation_targets.shape[1],
        bias=False,
    )

    with torch.no_grad():
        model.weight.copy_(
            torch.from_numpy(problem["base_weight"])
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-2,
        weight_decay=0.0,
    )

    initial_adaptation_loss = float(
        F.mse_loss(model(x), adaptation_targets).detach()
    )

    for _ in range(steps):
        prediction = model(x)
        loss = F.mse_loss(prediction, adaptation_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        final_adaptation_loss = float(
            F.mse_loss(model(x), adaptation_targets)
        )
        retained_base_loss = float(
            F.mse_loss(model(x), base_targets)
        )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "mode": "full",
        "steps": steps,
        "trainable_parameters": int(trainable),
        "initial_adaptation_loss": initial_adaptation_loss,
        "final_adaptation_loss": final_adaptation_loss,
        "retained_base_loss": retained_base_loss,
    }


def run_lora(
    *,
    steps: int = 20,
    rank: int = 2,
    alpha: float = 2.0,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0 or rank <= 0 or alpha <= 0:
        raise ValueError("steps/rank/alpha must be positive")

    torch.manual_seed(seed)
    problem = make_low_rank_adaptation_problem(
        rank=rank,
        seed=seed,
    )

    x = torch.from_numpy(problem["x"])
    base_targets = torch.from_numpy(problem["base_targets"])
    adaptation_targets = torch.from_numpy(
        problem["adaptation_targets"]
    )

    class LoRALinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            base = torch.from_numpy(
                problem["base_weight"]
            )
            self.base_weight = nn.Parameter(
                base.clone(),
                requires_grad=False,
            )

            self.a = nn.Parameter(
                torch.empty(
                    rank,
                    x.shape[1],
                )
            )
            self.b = nn.Parameter(
                torch.zeros(
                    adaptation_targets.shape[1],
                    rank,
                )
            )

            nn.init.normal_(
                self.a,
                mean=0.0,
                std=0.05,
            )

            self.scale = alpha / rank

        def forward(
            self,
            values: torch.Tensor,
        ) -> torch.Tensor:
            base_output = F.linear(
                values,
                self.base_weight,
            )
            adapter_output = (
                (values @ self.a.T)
                @ self.b.T
            )
            return (
                base_output
                + self.scale * adapter_output
            )

    model = LoRALinear()

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=5e-2,
        weight_decay=0.0,
    )

    initial_adaptation_loss = float(
        F.mse_loss(model(x), adaptation_targets).detach()
    )

    for _ in range(steps):
        prediction = model(x)
        loss = F.mse_loss(prediction, adaptation_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        final_adaptation_loss = float(
            F.mse_loss(model(x), adaptation_targets)
        )
        retained_base_loss = float(
            F.mse_loss(model(x), base_targets)
        )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    return {
        "mode": "lora",
        "steps": steps,
        "rank": rank,
        "alpha": alpha,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_fraction": float(trainable / total),
        "base_weight_requires_grad": bool(
            model.base_weight.requires_grad
        ),
        "initial_adaptation_loss": initial_adaptation_loss,
        "final_adaptation_loss": final_adaptation_loss,
        "retained_base_loss": retained_base_loss,
    }


def build_instruction_examples() -> list[dict[str, list[int]]]:
    instruction = _load_module(
        "57-instruction-tuning/src/instruction_data.py",
        "batch19_instruction_data",
    )

    raw = [
        ([1, 10, 11, 2], [20, 3]),
        ([1, 12, 13, 2], [21, 3]),
        ([1, 14, 15, 2], [22, 3]),
        ([1, 16, 17, 2], [23, 3]),
    ]

    examples = []

    for prompt, response in raw:
        ids, mask = instruction.concatenate_role_segments(
            [
                ("user", prompt),
                ("assistant", response),
            ]
        )
        labels = instruction.assistant_only_labels(
            ids,
            mask,
        )
        examples.append(
            {
                "input_ids": ids,
                "labels": labels,
            }
        )

    return examples


def run_instruction_sft(
    *,
    steps: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)

    examples = build_instruction_examples()
    input_ids = torch.tensor(
        [example["input_ids"] for example in examples],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [example["labels"] for example in examples],
        dtype=torch.long,
    )

    vocab_size = 32

    class TinyCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 16)
            self.rnn = nn.GRU(
                input_size=16,
                hidden_size=24,
                batch_first=True,
            )
            self.head = nn.Linear(24, vocab_size)

        def forward(
            self,
            token_ids: torch.Tensor,
        ) -> torch.Tensor:
            hidden, _ = self.rnn(
                self.embedding(token_ids)
            )
            return self.head(hidden)

    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    supervised_tokens = int(
        (labels != -100).sum().item()
    )
    losses = []

    for _ in range(steps):
        logits = model(input_ids)

        # Predict token t+1 from positions <= t.
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shifted_logits.view(-1, vocab_size),
            shifted_labels.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach()))

    return {
        "mode": "sft",
        "steps": steps,
        "examples": len(examples),
        "supervised_tokens": supervised_tokens,
        "total_tokens": int(labels.numel()),
        "supervised_fraction": float(
            supervised_tokens / labels.numel()
        ),
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 19 fine-tuning/LoRA/SFT lab."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "lora", "sft"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch19"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "full":
        report = run_full_finetune(
            steps=args.steps,
            seed=args.seed,
        )
    elif args.mode == "lora":
        report = run_lora(
            steps=args.steps,
            seed=args.seed,
        )
    else:
        report = run_instruction_sft(
            steps=args.steps,
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
