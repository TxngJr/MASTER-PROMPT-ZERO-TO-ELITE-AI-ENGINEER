"""Batch 15 integration: recommender, segmentation and vector retrieval."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def make_recommender_problem(
    *,
    users: int = 64,
    items: int = 96,
    latent_dim: int = 6,
    positives_per_user: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    if min(users, items, latent_dim, positives_per_user) <= 0:
        raise ValueError("dimensions must be positive")
    if positives_per_user >= items:
        raise ValueError("positives_per_user must be smaller than items")

    rng = np.random.default_rng(seed)
    user_latent = rng.normal(size=(users, latent_dim))
    item_latent = rng.normal(size=(items, latent_dim))
    scores = user_latent @ item_latent.T

    positives: list[tuple[int, int]] = []
    for user_id in range(users):
        top = np.argsort(-scores[user_id])[:positives_per_user]
        positives.extend(
            (user_id, int(item_id))
            for item_id in top
        )

    return (
        user_latent.astype(np.float32),
        item_latent.astype(np.float32),
        positives,
    )


def run_recommender(
    *,
    steps: int = 80,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F

    if steps <= 0:
        raise ValueError("steps must be positive")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_features, item_features, positives = make_recommender_problem(
        seed=seed
    )

    positives_by_user: dict[int, set[int]] = {}
    for user_id, item_id in positives:
        positives_by_user.setdefault(user_id, set()).add(item_id)

    user_tensor = torch.from_numpy(user_features).to(device)
    item_tensor = torch.from_numpy(item_features).to(device)

    class Tower(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 24),
                nn.GELU(),
                nn.Linear(24, 12),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.net(x), dim=-1)

    user_tower = Tower(user_features.shape[1]).to(device)
    item_tower = Tower(item_features.shape[1]).to(device)

    optimizer = torch.optim.AdamW(
        list(user_tower.parameters()) + list(item_tower.parameters()),
        lr=3e-3,
    )

    losses: list[float] = []

    for _ in range(steps):
        batch = rng.choice(len(positives), size=64, replace=True)

        user_ids = np.array(
            [positives[int(i)][0] for i in batch],
            dtype=np.int64,
        )
        positive_ids = np.array(
            [positives[int(i)][1] for i in batch],
            dtype=np.int64,
        )

        negative_ids = []
        for user_id in user_ids:
            while True:
                candidate = int(rng.integers(0, len(item_features)))
                if candidate not in positives_by_user[int(user_id)]:
                    negative_ids.append(candidate)
                    break

        user_z = user_tower(
            user_tensor[
                torch.as_tensor(user_ids, device=device)
            ]
        )
        positive_z = item_tower(
            item_tensor[
                torch.as_tensor(positive_ids, device=device)
            ]
        )
        negative_z = item_tower(
            item_tensor[
                torch.as_tensor(negative_ids, device=device)
            ]
        )

        positive_score = torch.sum(user_z * positive_z, dim=1)
        negative_score = torch.sum(user_z * negative_z, dim=1)
        loss = F.softplus(-(positive_score - negative_score)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    user_tower.eval()
    item_tower.eval()

    with torch.inference_mode():
        all_users = user_tower(user_tensor)
        all_items = item_tower(item_tensor)
        ranking_scores = all_users @ all_items.T

    recalls = []
    for user_id in range(len(user_features)):
        ranking = torch.argsort(
            ranking_scores[user_id],
            descending=True,
        ).cpu().numpy()

        relevant = positives_by_user[user_id]
        hits = sum(
            int(item_id) in relevant
            for item_id in ranking[:10]
        )
        recalls.append(hits / len(relevant))

    return {
        "mode": "recommender",
        "device": str(device),
        "steps": steps,
        "parameter_count": int(
            sum(p.numel() for p in user_tower.parameters())
            + sum(p.numel() for p in item_tower.parameters())
        ),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "mean_recall_at_10": float(np.mean(recalls)),
    }


def make_segmentation_dataset(
    n_samples: int = 96,
    *,
    image_size: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 0 or image_size < 8:
        raise ValueError("invalid dimensions")

    rng = np.random.default_rng(seed)
    images = np.zeros(
        (n_samples, 1, image_size, image_size),
        dtype=np.float32,
    )
    masks = np.zeros_like(images)

    for index in range(n_samples):
        width = int(rng.integers(5, 13))
        height = int(rng.integers(5, 13))
        x1 = int(rng.integers(1, image_size - width - 1))
        y1 = int(rng.integers(1, image_size - height - 1))
        x2 = x1 + width
        y2 = y1 + height

        masks[index, 0, y1:y2, x1:x2] = 1.0
        images[index] = masks[index]
        images[index] += rng.normal(
            0.0,
            0.12,
            size=images[index].shape,
        ).astype(np.float32)

    return images, masks


def run_segmentation(
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
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, masks = make_segmentation_dataset(seed=seed)
    split = 72

    x_train = torch.from_numpy(images[:split]).to(device)
    y_train = torch.from_numpy(masks[:split]).to(device)
    x_test = torch.from_numpy(images[split:]).to(device)
    y_test = torch.from_numpy(masks[split:]).to(device)

    class TinyUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = nn.Conv2d(1, 8, 3, padding=1)
            self.enc2 = nn.Conv2d(8, 16, 3, padding=1)
            self.dec1 = nn.Conv2d(24, 8, 3, padding=1)
            self.head = nn.Conv2d(8, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            skip = F.relu(self.enc1(x))
            pooled = F.max_pool2d(skip, 2)
            hidden = F.relu(self.enc2(pooled))
            up = F.interpolate(
                hidden,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded = F.relu(
                self.dec1(torch.cat([up, skip], dim=1))
            )
            return self.head(decoded)

    model = TinyUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses: list[float] = []

    for _ in range(steps):
        ids = rng.choice(split, size=24, replace=True)
        ids_t = torch.as_tensor(
            ids,
            dtype=torch.long,
            device=device,
        )
        xb = x_train[ids_t]
        yb = y_train[ids_t]

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()

    with torch.inference_mode():
        prediction = torch.sigmoid(model(x_test)) >= 0.5
        truth = y_test >= 0.5

        intersection = (prediction & truth).sum(
            dim=(1, 2, 3)
        ).float()
        union = (prediction | truth).sum(
            dim=(1, 2, 3)
        ).float().clamp_min(1.0)
        mean_iou = float(
            (intersection / union).mean().cpu()
        )

    return {
        "mode": "segmentation",
        "device": str(device),
        "steps": steps,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "test_mean_iou": mean_iou,
    }


def _load_vector_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root
        / "45-embeddings-vector-databases"
        / "src"
        / "vector_index_numpy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "batch15_vector_index",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_vector_search(
    *,
    seed: int = 42,
) -> dict[str, Any]:
    vector_mod = _load_vector_module()
    rng = np.random.default_rng(seed)

    centers = rng.normal(size=(8, 16))
    cluster_ids = rng.integers(0, len(centers), size=600)
    vectors = (
        centers[cluster_ids]
        + rng.normal(0.0, 0.15, size=(600, 16))
    )

    queries = vectors[
        rng.choice(len(vectors), size=30, replace=False)
    ]

    index = vector_mod.IVFIndex(
        clusters=8,
        iterations=8,
        seed=seed,
    ).fit(vectors)

    recalls_one = []
    recalls_four = []

    for query in queries:
        exact_ids, _ = vector_mod.exact_top_k(
            query,
            vectors,
            10,
        )
        ids_one, _ = index.search(
            query,
            10,
            nprobe=1,
        )
        ids_four, _ = index.search(
            query,
            10,
            nprobe=4,
        )

        recalls_one.append(
            vector_mod.ann_recall_at_k(
                exact_ids,
                ids_one,
                10,
            )
        )
        recalls_four.append(
            vector_mod.ann_recall_at_k(
                exact_ids,
                ids_four,
                10,
            )
        )

    return {
        "mode": "vector-search",
        "vectors": int(len(vectors)),
        "dimension": int(vectors.shape[1]),
        "mean_recall_at_10_nprobe_1": float(
            np.mean(recalls_one)
        ),
        "mean_recall_at_10_nprobe_4": float(
            np.mean(recalls_four)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 15 recommender/vision/vector lab."
    )
    parser.add_argument(
        "--mode",
        choices=["recommender", "segmentation", "vector-search"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch15"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "recommender":
        report = run_recommender(
            steps=args.steps,
            seed=args.seed,
        )
    elif args.mode == "segmentation":
        report = run_segmentation(
            steps=args.steps,
            seed=args.seed,
        )
    else:
        report = run_vector_search(seed=args.seed)

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
