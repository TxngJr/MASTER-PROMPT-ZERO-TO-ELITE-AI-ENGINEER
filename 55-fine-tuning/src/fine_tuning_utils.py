"""Framework-independent fine-tuning planning utilities."""

from __future__ import annotations

from dataclasses import dataclass


def parameter_partition(
    named_parameter_sizes: dict[str, int],
    trainable_names: set[str],
) -> tuple[int, int]:
    if any(size < 0 for size in named_parameter_sizes.values()):
        raise ValueError("parameter sizes must be non-negative")

    total = sum(named_parameter_sizes.values())
    trainable = sum(
        size
        for name, size in named_parameter_sizes.items()
        if name in trainable_names
    )
    return trainable, total


def trainable_ratio(
    trainable_parameters: int,
    total_parameters: int,
) -> float:
    if trainable_parameters < 0:
        raise ValueError("trainable_parameters must be non-negative")
    if total_parameters <= 0:
        raise ValueError("total_parameters must be positive")
    if trainable_parameters > total_parameters:
        raise ValueError("trainable cannot exceed total")

    return float(trainable_parameters / total_parameters)


def freeze_by_prefix(
    parameter_names: list[str],
    prefixes: tuple[str, ...],
) -> set[str]:
    """Return names that remain trainable after freezing prefixes."""
    return {
        name
        for name in parameter_names
        if not name.startswith(prefixes)
    }


def unfreeze_by_prefix(
    parameter_names: list[str],
    prefixes: tuple[str, ...],
) -> set[str]:
    """Return only names whose prefix should be trainable."""
    return {
        name
        for name in parameter_names
        if name.startswith(prefixes)
    }


def discriminative_lr_groups(
    parameter_names: list[str],
    *,
    rules: list[tuple[str, float]],
    default_lr: float,
) -> dict[str, float]:
    if default_lr <= 0:
        raise ValueError("default_lr must be positive")
    if any(lr <= 0 for _, lr in rules):
        raise ValueError("all learning rates must be positive")

    result: dict[str, float] = {}

    for name in parameter_names:
        learning_rate = default_lr

        for prefix, candidate_lr in rules:
            if name.startswith(prefix):
                learning_rate = candidate_lr
                break

        result[name] = float(learning_rate)

    return result


@dataclass(frozen=True)
class EarlyStoppingState:
    best_metric: float | None = None
    bad_epochs: int = 0
    should_stop: bool = False


def early_stopping_update(
    state: EarlyStoppingState,
    metric: float,
    *,
    patience: int,
    mode: str = "min",
    min_delta: float = 0.0,
) -> EarlyStoppingState:
    if patience < 0:
        raise ValueError("patience must be non-negative")
    if mode not in {"min", "max"}:
        raise ValueError("mode must be min or max")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")

    if state.best_metric is None:
        improved = True
    elif mode == "min":
        improved = metric < state.best_metric - min_delta
    else:
        improved = metric > state.best_metric + min_delta

    if improved:
        return EarlyStoppingState(
            best_metric=float(metric),
            bad_epochs=0,
            should_stop=False,
        )

    bad_epochs = state.bad_epochs + 1
    return EarlyStoppingState(
        best_metric=state.best_metric,
        bad_epochs=bad_epochs,
        should_stop=bad_epochs > patience,
    )


def select_best_checkpoint(
    metrics: list[float],
    *,
    mode: str = "min",
) -> int:
    if not metrics:
        raise ValueError("metrics must be non-empty")
    if mode not in {"min", "max"}:
        raise ValueError("mode must be min or max")

    if mode == "min":
        return min(range(len(metrics)), key=metrics.__getitem__)
    return max(range(len(metrics)), key=metrics.__getitem__)
