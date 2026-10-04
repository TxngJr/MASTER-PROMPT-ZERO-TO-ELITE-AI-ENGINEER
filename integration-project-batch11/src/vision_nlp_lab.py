"""Batch 11 integration: Tiny ViT and static embedding experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def _load_embedding_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root
        / "33-word-embeddings"
        / "src"
        / "word_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "batch11_word_embeddings",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_embedding_lab(
    *,
    epochs: int = 4,
    embedding_dim: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    if epochs <= 0 or embedding_dim <= 0:
        raise ValueError("epochs and embedding_dim must be positive")

    emb = _load_embedding_module()

    sentences = [
        "cat pet animal",
        "dog pet animal",
        "cat sleeps home",
        "dog sleeps home",
        "python code programming",
        "rust code programming",
        "python software language",
        "rust software language",
        "apple fruit sweet",
        "banana fruit sweet",
    ] * 12

    tokenized = [sentence.split() for sentence in sentences]
    words = sorted({word for row in tokenized for word in row})
    stoi = {word: index for index, word in enumerate(words)}
    ids = [stoi[word] for row in tokenized for word in row]

    pairs = emb.skipgram_pairs(ids, window_size=2)
    distribution = emb.negative_sampling_distribution(
        ids,
        vocab_size=len(words),
    )

    rng = np.random.default_rng(seed)
    skipgram = emb.SkipGramNegativeSampling(
        vocab_size=len(words),
        embedding_dim=embedding_dim,
        learning_rate=0.05,
        seed=seed,
    )

    skipgram_losses = []
    for _ in range(epochs):
        order = rng.permutation(len(pairs))
        running = 0.0

        for pair_index in order:
            center, context = pairs[pair_index]
            negatives = rng.choice(
                len(words),
                size=4,
                replace=True,
                p=distribution,
            )
            running += skipgram.train_pair(
                center,
                context,
                negatives,
            )

        skipgram_losses.append(running / len(pairs))

    cooc = emb.cooccurrence_matrix(
        ids,
        vocab_size=len(words),
        window_size=2,
    )
    glove = emb.GloVeModel(
        vocab_size=len(words),
        embedding_dim=embedding_dim,
        learning_rate=0.03,
        x_max=20.0,
        seed=seed,
    )

    nonzero = np.argwhere(cooc > 0)
    glove_losses = []

    for _ in range(epochs):
        running = 0.0
        for i, j in nonzero:
            running += glove.train_entry(
                int(i),
                int(j),
                float(cooc[i, j]),
            )
        glove_losses.append(running / max(1, len(nonzero)))

    fasttext = emb.FastTextSubwordTable(
        embedding_dim=embedding_dim,
        min_n=3,
        max_n=5,
        seed=seed,
    ).fit_vocabulary(words)

    def similarity(
        matrix: np.ndarray,
        left: str,
        right: str,
    ) -> float:
        return emb.cosine_similarity(
            matrix[stoi[left]],
            matrix[stoi[right]],
        )

    skip_matrix = (
        skipgram.input_embeddings
        + skipgram.output_embeddings
    )
    glove_matrix = glove.combined_embeddings()

    return {
        "mode": "embeddings",
        "vocabulary_size": len(words),
        "pair_count": len(pairs),
        "skipgram_loss_history": [
            float(value) for value in skipgram_losses
        ],
        "glove_loss_history": [
            float(value) for value in glove_losses
        ],
        "skipgram_similarity": {
            "cat_dog": similarity(skip_matrix, "cat", "dog"),
            "python_rust": similarity(skip_matrix, "python", "rust"),
        },
        "glove_similarity": {
            "cat_dog": similarity(glove_matrix, "cat", "dog"),
            "python_rust": similarity(glove_matrix, "python", "rust"),
        },
        "fasttext_oov_norm": float(
            np.linalg.norm(fasttext.vector("programminglike"))
        ),
    }


def _digits_split(
    *,
    seed: int,
    limit: int | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = load_digits()
    images = data.images.astype(np.float32) / 16.0
    labels = data.target.astype(np.int64)

    if limit is not None and limit < len(images):
        if limit < 200:
            raise ValueError("limit must be at least 200")
        _, images, _, labels = train_test_split(
            images,
            labels,
            test_size=limit,
            stratify=labels,
            random_state=seed,
        )

    x_train, x_temp, y_train, y_temp = train_test_split(
        images,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )

    return {
        "train": (x_train, y_train),
        "validation": (x_val, y_val),
        "test": (x_test, y_test),
    }


def run_vit(
    *,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
    limit: int | None = None,
) -> dict[str, Any]:
    import copy

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    splits = _digits_split(seed=seed, limit=limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TinyViT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            model_dim = 32
            patch_size = 2
            patch_count = (8 // patch_size) ** 2

            self.patch_embed = nn.Conv2d(
                1,
                model_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, model_dim)
            )
            self.position = nn.Parameter(
                torch.zeros(1, patch_count + 1, model_dim)
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=4,
                dim_feedforward=64,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
            )
            self.norm = nn.LayerNorm(model_dim)
            self.head = nn.Linear(model_dim, 10)

            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.position, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)

            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + self.position[:, : x.shape[1], :]

            x = self.encoder(x)
            return self.head(self.norm(x[:, 0]))

    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]
    x_test, y_test = splits["test"]

    train_dataset = TensorDataset(
        torch.from_numpy(x_train[:, None, :, :]),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = TinyViT().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(
                torch.from_numpy(
                    x_val[:, None, :, :]
                ).to(device)
            )
            val_pred = torch.argmax(
                val_logits,
                dim=1,
            ).cpu().numpy()

        val_accuracy = float(np.mean(val_pred == y_val))

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("no ViT checkpoint produced")

    model.load_state_dict(best_state)
    model.eval()

    with torch.inference_mode():
        test_logits = model(
            torch.from_numpy(
                x_test[:, None, :, :]
            ).to(device)
        )
        test_pred = torch.argmax(
            test_logits,
            dim=1,
        ).cpu().numpy()

    return {
        "mode": "vit",
        "device": str(device),
        "epochs": epochs,
        "parameter_count": int(
            sum(p.numel() for p in model.parameters())
        ),
        "patch_size": 2,
        "patch_count": 16,
        "best_validation_accuracy": float(best_val),
        "test_accuracy": float(np.mean(test_pred == y_test)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 11 ViT/static-embedding lab."
    )
    parser.add_argument(
        "--mode",
        choices=["vit", "embeddings"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch11"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "vit":
        report = run_vit(
            epochs=args.epochs,
            seed=args.seed,
            limit=args.limit,
        )
    else:
        report = run_embedding_lab(
            epochs=args.epochs,
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
