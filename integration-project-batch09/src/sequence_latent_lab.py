"""Batch 09 PyTorch integration for recurrent models and a tiny VAE."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def make_memory_dataset(
    n_samples: int = 1200,
    *,
    sequence_length: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 200:
        raise ValueError("n_samples must be at least 200")
    if sequence_length < 5:
        raise ValueError("sequence_length must be at least 5")

    rng = np.random.default_rng(seed)
    first_signal = rng.integers(0, 2, size=n_samples)
    x = rng.normal(0.0, 1.0, size=(n_samples, sequence_length, 1))
    x[:, 0, 0] = np.where(first_signal == 1, 2.0, -2.0)
    y = first_signal.astype(np.int64)
    return x.astype(np.float32), y


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )
    return {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
    }


def run_sequence_benchmark(
    *,
    epochs: int = 5,
    hidden_size: int = 24,
    batch_size: int = 64,
    seed: int = 42,
    limit: int | None = None,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x, y = make_memory_dataset(
        n_samples=1200 if limit is None else max(200, limit),
        seed=seed,
    )
    splits = stratified_split(x, y, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SequenceClassifier(nn.Module):
        def __init__(self, kind: str) -> None:
            super().__init__()
            recurrent_cls = {
                "rnn": nn.RNN,
                "lstm": nn.LSTM,
                "gru": nn.GRU,
            }[kind]
            self.kind = kind
            self.recurrent = recurrent_cls(
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 2)

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            _, state = self.recurrent(xb)
            if self.kind == "lstm":
                hidden, _ = state
            else:
                hidden = state
            final = hidden[-1]
            return self.head(final)

    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]
    x_test, y_test = splits["test"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    results: dict[str, Any] = {}

    for kind in ["rnn", "lstm", "gru"]:
        torch.manual_seed(seed)
        model = SequenceClassifier(kind).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        best_val = -1.0
        best_state = None
        started = time.perf_counter()

        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            with torch.inference_mode():
                val_logits = model(torch.from_numpy(x_val).to(device))
                val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_accuracy = float(np.mean(val_pred == y_val))

            if val_accuracy > best_val:
                best_val = val_accuracy
                best_state = copy.deepcopy(model.state_dict())

        elapsed = time.perf_counter() - started

        if best_state is None:
            raise RuntimeError("no recurrent checkpoint produced")
        model.load_state_dict(best_state)
        model.eval()

        with torch.inference_mode():
            test_logits = model(torch.from_numpy(x_test).to(device))
            test_pred = torch.argmax(test_logits, dim=1).cpu().numpy()

        results[kind] = {
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "best_validation_accuracy": float(best_val),
            "test_accuracy": float(np.mean(test_pred == y_test)),
            "train_seconds": float(elapsed),
        }

    selected = max(
        results,
        key=lambda name: results[name]["best_validation_accuracy"],
    )

    return {
        "mode": "sequence",
        "device": str(device),
        "seed": seed,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "results": results,
        "selected_model": selected,
    }


def run_vae(
    *,
    epochs: int = 5,
    latent_dim: int = 8,
    batch_size: int = 64,
    beta: float = 1.0,
    seed: int = 42,
    limit: int | None = None,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    digits = load_digits()
    x = (digits.data.astype(np.float32) / 16.0)
    if limit is not None and limit < len(x):
        x, _ = train_test_split(
            x,
            train_size=limit,
            random_state=seed,
        )

    x_train, x_temp = train_test_split(
        x,
        test_size=0.30,
        random_state=seed,
    )
    x_val, x_test = train_test_split(
        x_temp,
        test_size=0.50,
        random_state=seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TinyVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.mu = nn.Linear(32, latent_dim)
            self.logvar = nn.Linear(32, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
            )

        def encode(self, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder(xb)
            return self.mu(h), self.logvar(h)

        def reparameterize(
            self,
            mu: torch.Tensor,
            logvar: torch.Tensor,
        ) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def forward(
            self,
            xb: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(xb)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar

    def loss_components(
        logits: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="sum",
        ) / target.shape[0]
        kl = (
            -0.5
            * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
            / target.shape[0]
        )
        return recon + beta * kl, recon, kl

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    model = TinyVAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    history: list[dict[str, float]] = []

    for _ in range(epochs):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = model(xb)
            total, _, _ = loss_components(logits, xb, mu, logvar)
            total.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val = torch.from_numpy(x_val).to(device)
            logits, mu, logvar = model(val)
            total, recon, kl = loss_components(logits, val, mu, logvar)
            history.append(
                {
                    "total": float(total.cpu()),
                    "reconstruction": float(recon.cpu()),
                    "kl": float(kl.cpu()),
                }
            )

    model.eval()
    with torch.inference_mode():
        test = torch.from_numpy(x_test).to(device)
        logits, mu, logvar = model(test)
        total, recon, kl = loss_components(logits, test, mu, logvar)

        z_prior = torch.randn(16, latent_dim, device=device)
        samples = torch.sigmoid(model.decoder(z_prior))

    return {
        "mode": "vae",
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "latent_dim": latent_dim,
        "beta": beta,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "validation_history": history,
        "test": {
            "total": float(total.cpu()),
            "reconstruction": float(recon.cpu()),
            "kl": float(kl.cpu()),
            "mu_mean": float(mu.mean().cpu()),
            "mu_std": float(mu.std().cpu()),
            "logvar_mean": float(logvar.mean().cpu()),
        },
        "prior_samples": {
            "min": float(samples.min().cpu()),
            "max": float(samples.max().cpu()),
            "mean": float(samples.mean().cpu()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Batch 09 integration lab.")
    parser.add_argument("--mode", choices=["sequence", "vae"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch09"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "sequence":
        report = run_sequence_benchmark(
            epochs=args.epochs,
            seed=args.seed,
            limit=args.limit,
        )
    else:
        report = run_vae(
            epochs=args.epochs,
            seed=args.seed,
            limit=args.limit,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.mode}.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"mode: {report['mode']}")
    print(f"device: {report['device']}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
