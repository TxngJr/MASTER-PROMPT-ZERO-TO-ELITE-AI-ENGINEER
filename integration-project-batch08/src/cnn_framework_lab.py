"""Train equivalent tiny CNNs on sklearn digits with PyTorch or TensorFlow."""

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


def load_digits_splits(
    *,
    seed: int = 42,
    limit: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = load_digits()
    images = data.images.astype(np.float32) / 16.0
    labels = data.target.astype(np.int64)

    if limit is not None:
        if limit < 100:
            raise ValueError("limit must be at least 100")
        if limit < len(images):
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


def to_nchw(images: np.ndarray) -> np.ndarray:
    x = np.asarray(images, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("expected images with shape (N,H,W)")
    return x[:, None, :, :]


def to_nhwc(images: np.ndarray) -> np.ndarray:
    x = np.asarray(images, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("expected images with shape (N,H,W)")
    return x[:, :, :, None]


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def run_pytorch(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TinyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Linear(16 * 2 * 2, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = torch.flatten(x, start_dim=1)
            return self.classifier(x)

    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]
    x_test, y_test = splits["test"]

    train_dataset = TensorDataset(
        torch.from_numpy(to_nchw(x_train)),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = TinyCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None

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
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(
                torch.from_numpy(to_nchw(x_val)).to(device)
            )
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()

        val_accuracy = _accuracy(y_val, val_pred)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    train_seconds = time.perf_counter() - started

    if best_state is None:
        raise RuntimeError("no PyTorch checkpoint was produced")

    model.load_state_dict(best_state)
    model.eval()

    predict_started = time.perf_counter()
    with torch.inference_mode():
        test_logits = model(
            torch.from_numpy(to_nchw(x_test)).to(device)
        )
        test_pred = torch.argmax(test_logits, dim=1).cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()

    predict_seconds = time.perf_counter() - predict_started

    parameter_count = sum(p.numel() for p in model.parameters())

    return {
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "device": str(device),
        "parameter_count": int(parameter_count),
        "best_validation_accuracy": float(best_val),
        "test_accuracy": _accuracy(y_test, test_pred),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
    }


def run_tensorflow(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    import tensorflow as tf
    from tensorflow import keras

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    keras.utils.set_random_seed(seed)

    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]
    x_test, y_test = splits["test"]

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(8, 8, 1)),
            keras.layers.Conv2D(
                8,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(
                16,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    callback = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=max(1, min(3, epochs)),
        restore_best_weights=True,
    )

    started = time.perf_counter()
    history = model.fit(
        to_nhwc(x_train),
        y_train,
        validation_data=(to_nhwc(x_val), y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[callback],
    )
    train_seconds = time.perf_counter() - started

    val_history = history.history.get("val_accuracy", [])
    best_val = max(float(v) for v in val_history) if val_history else 0.0

    predict_started = time.perf_counter()
    logits = model.predict(to_nhwc(x_test), batch_size=batch_size, verbose=0)
    test_pred = np.argmax(logits, axis=1)
    predict_seconds = time.perf_counter() - predict_started

    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

    return {
        "framework": "tensorflow",
        "framework_version": tf.__version__,
        "device": device,
        "parameter_count": int(model.count_params()),
        "best_validation_accuracy": float(best_val),
        "test_accuracy": _accuracy(y_test, test_pred),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
    }


def run_experiment(
    framework: str,
    *,
    seed: int = 42,
    epochs: int = 5,
    batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    splits = load_digits_splits(seed=seed, limit=limit)

    if framework == "pytorch":
        result = run_pytorch(
            splits,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
    elif framework == "tensorflow":
        result = run_tensorflow(
            splits,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        raise ValueError("framework must be 'pytorch' or 'tensorflow'")

    result["seed"] = seed
    result["epochs_requested"] = epochs
    result["batch_size"] = batch_size
    result["split_sizes"] = {
        name: int(len(values[1]))
        for name, values in splits.items()
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Batch 08 cross-framework CNN lab."
    )
    parser.add_argument(
        "--framework",
        choices=["pytorch", "tensorflow"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/batch08"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report = run_experiment(
        args.framework,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.framework}.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"framework: {report['framework']}")
    print(f"device: {report['device']}")
    print(f"test accuracy: {report['test_accuracy']:.4f}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
