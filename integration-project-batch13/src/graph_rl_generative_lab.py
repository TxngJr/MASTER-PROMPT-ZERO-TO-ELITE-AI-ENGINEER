"""Batch 13 integration: GNN, DQN and generative-distribution labs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def make_community_graph(
    *,
    nodes_per_class: int = 40,
    classes: int = 3,
    feature_dim: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(nodes_per_class, classes, feature_dim) <= 0:
        raise ValueError("dimensions must be positive")

    rng = np.random.default_rng(seed)
    num_nodes = nodes_per_class * classes

    labels = np.repeat(np.arange(classes), nodes_per_class)

    centers = rng.normal(
        0.0,
        2.0,
        size=(classes, feature_dim),
    )
    features = centers[labels] + rng.normal(
        0.0,
        0.7,
        size=(num_nodes, feature_dim),
    )

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for class_id in range(classes):
        start = class_id * nodes_per_class
        end = start + nodes_per_class

        for node in range(start, end):
            # Ring edges guarantee connectivity within each community.
            next_node = start + ((node - start + 1) % nodes_per_class)
            adjacency[node, next_node] = 1.0
            adjacency[next_node, node] = 1.0

            # Add a few extra same-class edges.
            candidates = np.arange(start, end)
            extras = rng.choice(candidates, size=3, replace=False)
            adjacency[node, extras] = 1.0
            adjacency[extras, node] = 1.0

    # Small amount of cross-community noise.
    for _ in range(max(1, num_nodes // 8)):
        a = int(rng.integers(0, num_nodes))
        b = int(rng.integers(0, num_nodes))
        if labels[a] != labels[b]:
            adjacency[a, b] = 1.0
            adjacency[b, a] = 1.0

    np.fill_diagonal(adjacency, 0.0)

    return (
        features.astype(np.float32),
        adjacency,
        labels.astype(np.int64),
    )


def _normalized_adjacency(adjacency: np.ndarray) -> np.ndarray:
    a = np.asarray(adjacency, dtype=np.float32).copy()
    a = a + np.eye(len(a), dtype=np.float32)
    degree = a.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


def run_gnn(
    *,
    epochs: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, adjacency, labels = make_community_graph(seed=seed)
    normalized = _normalized_adjacency(adjacency)

    x = torch.from_numpy(features).to(device)
    a = torch.from_numpy(normalized).to(device)
    y = torch.from_numpy(labels).to(device)

    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)

    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        n = len(indices)
        train_mask[indices[: int(0.6 * n)]] = True
        val_mask[indices[int(0.6 * n) : int(0.8 * n)]] = True
        test_mask[indices[int(0.8 * n) :]] = True

    train_mask_t = torch.from_numpy(train_mask).to(device)
    val_mask_t = torch.from_numpy(val_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)

    class TinyGCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(x.shape[1], 16, bias=False)
            self.linear2 = nn.Linear(16, 3, bias=False)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            h = a @ features
            h = F.relu(self.linear1(h))
            h = a @ h
            return self.linear2(h)

    model = TinyGCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

    best_val = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(
            logits[train_mask_t],
            y[train_mask_t],
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            val_accuracy = float(
                (pred[val_mask_t] == y[val_mask_t])
                .float()
                .mean()
                .cpu()
            )

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no GCN checkpoint produced")

    model.load_state_dict(best_state)
    model.eval()

    with torch.inference_mode():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        test_accuracy = float(
            (pred[test_mask_t] == y[test_mask_t])
            .float()
            .mean()
            .cpu()
        )

    return {
        "mode": "gnn",
        "device": str(device),
        "epochs": epochs,
        "nodes": int(len(labels)),
        "edges": int(adjacency.sum() // 2),
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "best_validation_accuracy": float(best_val),
        "test_accuracy": test_accuracy,
    }


class ChainEnv:
    def __init__(self, length: int = 7) -> None:
        if length < 3:
            raise ValueError("length must be at least 3")
        self.length = length
        self.state = 0

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")

        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.length - 1, self.state + 1)

        done = self.state == self.length - 1
        reward = 1.0 if done else -0.02
        return self.state, reward, done


def run_dqn(
    *,
    episodes: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    import random

    import torch
    from torch import nn
    from torch.nn import functional as F

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    random.seed(seed)
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ChainEnv(length=7)

    class QNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(env.length, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state)

    online = QNetwork().to(device)
    target = QNetwork().to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=0.01)

    replay: list[tuple[int, int, float, int, bool]] = []
    returns = []
    successes = 0

    def one_hot(states: list[int] | np.ndarray) -> torch.Tensor:
        values = torch.as_tensor(
            states,
            dtype=torch.long,
            device=device,
        )
        return F.one_hot(
            values,
            num_classes=env.length,
        ).float()

    for episode in range(episodes):
        state = env.reset()
        episode_return = 0.0
        epsilon = max(0.05, 0.8 * (0.95 ** episode))

        for _ in range(30):
            if np_rng.random() < epsilon:
                action = int(np_rng.integers(0, 2))
            else:
                online.eval()
                with torch.inference_mode():
                    q = online(one_hot([state]))[0]
                    action = int(torch.argmax(q).cpu())

            next_state, reward, done = env.step(action)
            replay.append((state, action, reward, next_state, done))
            if len(replay) > 500:
                replay.pop(0)

            state = next_state
            episode_return += reward

            if len(replay) >= 16:
                batch_ids = np_rng.choice(
                    len(replay),
                    size=16,
                    replace=False,
                )
                batch = [replay[int(i)] for i in batch_ids]

                states = one_hot([item[0] for item in batch])
                actions = torch.tensor(
                    [item[1] for item in batch],
                    dtype=torch.long,
                    device=device,
                )
                rewards = torch.tensor(
                    [item[2] for item in batch],
                    dtype=torch.float32,
                    device=device,
                )
                next_states = one_hot([item[3] for item in batch])
                dones = torch.tensor(
                    [item[4] for item in batch],
                    dtype=torch.float32,
                    device=device,
                )

                online.train()
                q_values = online(states).gather(
                    1,
                    actions[:, None],
                ).squeeze(1)

                with torch.no_grad():
                    next_q = target(next_states).max(dim=1).values
                    td_target = rewards + 0.95 * (1.0 - dones) * next_q

                loss = F.huber_loss(q_values, td_target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if done:
                successes += 1
                break

        returns.append(episode_return)

        if (episode + 1) % 5 == 0:
            target.load_state_dict(online.state_dict())

    return {
        "mode": "dqn",
        "device": str(device),
        "episodes": episodes,
        "parameter_count": int(
            sum(p.numel() for p in online.parameters())
        ),
        "success_rate": float(successes / episodes),
        "mean_return_last_10": float(np.mean(returns[-10:])),
        "replay_size": int(len(replay)),
    }


def make_multimodal_points(
    n_samples: int = 800,
    *,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [-2.0, -2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    choices = rng.integers(0, len(centers), size=n_samples)
    return (
        centers[choices]
        + rng.normal(0.0, 0.35, size=(n_samples, 2))
    ).astype(np.float64)


def run_generative(
    *,
    seed: int = 42,
) -> dict[str, Any]:
    points = make_multimodal_points(seed=seed)

    mean = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    covariance += np.eye(2) * 1e-6

    inverse = np.linalg.inv(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise RuntimeError("covariance is not positive definite")

    centered = points - mean
    quadratic = np.einsum(
        "ni,ij,nj->n",
        centered,
        inverse,
        centered,
    )
    nll = 0.5 * (
        2 * np.log(2.0 * np.pi)
        + logdet
        + quadratic
    )

    rng = np.random.default_rng(seed + 1)
    samples = rng.multivariate_normal(
        mean,
        covariance,
        size=400,
    )

    alpha_bars = [1.0, 0.7, 0.3, 0.0]
    noising = {}

    probe = points[:100]

    for alpha_bar in alpha_bars:
        noise = rng.normal(size=probe.shape)
        noisy = (
            np.sqrt(alpha_bar) * probe
            + np.sqrt(1.0 - alpha_bar) * noise
        )
        noising[str(alpha_bar)] = {
            "mean_norm": float(
                np.mean(np.linalg.norm(noisy, axis=1))
            ),
            "correlation_with_clean": float(
                np.corrcoef(
                    probe.reshape(-1),
                    noisy.reshape(-1),
                )[0, 1]
            ) if alpha_bar > 0.0 else 0.0,
        }

    return {
        "mode": "generative",
        "data_mean": mean.tolist(),
        "data_covariance": covariance.tolist(),
        "single_gaussian_mean_nll": float(np.mean(nll)),
        "sample_mean": samples.mean(axis=0).tolist(),
        "sample_covariance": np.cov(samples, rowvar=False).tolist(),
        "forward_noising": noising,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 13 GNN/RL/generative lab."
    )
    parser.add_argument(
        "--mode",
        choices=["gnn", "dqn", "generative"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch13"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "gnn":
        report = run_gnn(epochs=args.steps, seed=args.seed)
    elif args.mode == "dqn":
        report = run_dqn(episodes=args.steps, seed=args.seed)
    else:
        report = run_generative(seed=args.seed)

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
