"""Small PyTorch utilities used by Chapter 22."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def choose_device(*, prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TinyClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if min(in_features, hidden_features, num_classes) <= 0:
            raise ValueError("feature/class counts must be positive")

        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parameter_count(model: nn.Module, *, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


@torch.inference_mode()
def predict_classes(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return torch.argmax(model(x), dim=1)
