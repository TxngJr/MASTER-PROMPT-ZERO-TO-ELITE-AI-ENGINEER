"""Train a tiny MLP using the course's own autodiff and optimizer code."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tensor_mod = _load(
    "course_tensor",
    ROOT / "19-backpropagation" / "src" / "tensor.py",
)
optimizer_mod = _load(
    "course_optimizers",
    ROOT / "20-optimizers" / "src" / "optimizers.py",
)
loss_mod = _load(
    "course_losses",
    ROOT / "21-activations-losses" / "src" / "activations_losses.py",
)

Tensor = tensor_mod.Tensor
AdamW = optimizer_mod.AdamW


class TinyMLPClassifier:
    def __init__(
        self,
        *,
        hidden_features: int = 16,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        self.w1 = Tensor(
            rng.normal(0.0, np.sqrt(1.0 / 2.0), size=(2, hidden_features)),
            requires_grad=True,
        )
        self.b1 = Tensor(
            np.zeros(hidden_features),
            requires_grad=True,
        )
        self.w2 = Tensor(
            rng.normal(
                0.0,
                np.sqrt(1.0 / hidden_features),
                size=(hidden_features, 1),
            ),
            requires_grad=True,
        )
        self.b2 = Tensor(
            np.zeros(1),
            requires_grad=True,
        )

    def parameters(self) -> list[Tensor]:
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, X: np.ndarray) -> Tensor:
        inputs = Tensor(np.asarray(X, dtype=float))
        hidden = ((inputs @ self.w1) + self.b1).tanh()
        return (hidden @ self.w2) + self.b2

    def snapshot(self) -> list[np.ndarray]:
        return [parameter.data.copy() for parameter in self.parameters()]

    def restore(self, values: list[np.ndarray]) -> None:
        for parameter, value in zip(self.parameters(), values):
            parameter.data = value.copy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X).data.reshape(-1)
        return loss_mod.stable_sigmoid(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_dataset(
    n_samples: int = 600,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 200:
        raise ValueError("n_samples must be at least 200")
    X, y = make_moons(
        n_samples=n_samples,
        noise=0.16,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )

    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_validation),
        scaler.transform(X_test),
        y_train,
        y_validation,
        y_test,
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 900,
) -> tuple[TinyMLPClassifier, dict[str, Any]]:
    model = TinyMLPClassifier(seed=seed)
    optimizer = AdamW(
        model.parameters(),
        lr=0.03,
        weight_decay=1e-4,
    )

    y_train_column = y_train.astype(float).reshape(-1, 1)
    y_validation_column = y_validation.astype(float).reshape(-1, 1)

    history: list[float] = []
    validation_history: list[float] = []
    best_validation = float("inf")
    best_snapshot = model.snapshot()
    best_epoch = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = model.forward(X_train)
        loss, gradient = loss_mod.bce_with_logits_loss_and_grad(
            logits.data,
            y_train_column,
        )

        logits.backward(gradient)
        optimizer.step()
        history.append(loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            validation_logits = model.forward(X_validation).data
            validation_loss, _ = loss_mod.bce_with_logits_loss_and_grad(
                validation_logits,
                y_validation_column,
            )
            validation_history.append(validation_loss)

            if validation_loss < best_validation:
                best_validation = validation_loss
                best_snapshot = model.snapshot()
                best_epoch = epoch

    model.restore(best_snapshot)

    return model, {
        "train_loss_initial": float(history[0]),
        "train_loss_final": float(history[-1]),
        "best_validation_loss": float(best_validation),
        "best_epoch": int(best_epoch),
        "epochs": int(epochs),
        "validation_checks": int(len(validation_history)),
    }


def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 900,
) -> dict[str, Any]:
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_and_scale(X, y, seed=seed)

    model, training = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        seed=seed,
        epochs=epochs,
    )

    validation_prediction = model.predict(X_validation)
    test_prediction = model.predict(X_test)

    return {
        "seed": seed,
        "rows": int(len(y)),
        "architecture": [2, 16, 1],
        "parameters": int(
            sum(parameter.data.size for parameter in model.parameters())
        ),
        "training": training,
        "validation": {
            "accuracy": float(
                accuracy_score(y_validation, validation_prediction)
            ),
            "f1": float(f1_score(y_validation, validation_prediction)),
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, test_prediction)),
            "f1": float(f1_score(y_test, test_prediction)),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Batch 07 tiny autodiff MLP."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch07"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    X, y = make_dataset(args.samples, seed=args.seed)
    report = run_experiment(
        X,
        y,
        seed=args.seed,
        epochs=args.epochs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"test accuracy: {report['test']['accuracy']:.4f}")
    print(f"test F1: {report['test']['f1']:.4f}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
