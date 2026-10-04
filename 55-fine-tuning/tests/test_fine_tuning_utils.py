from pathlib import Path
import importlib.util
import sys


MODULE_PATH = Path(__file__).parents[1] / "src" / "fine_tuning_utils.py"
SPEC = importlib.util.spec_from_file_location("fine_tuning_utils_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_trainable_ratio() -> None:
    trainable, total = mod.parameter_partition(
        {"backbone.a": 100, "head.weight": 20},
        {"head.weight"},
    )
    assert trainable == 20
    assert total == 120
    assert mod.trainable_ratio(trainable, total) == 1 / 6


def test_freeze_and_unfreeze_prefix() -> None:
    names = [
        "encoder.0.weight",
        "encoder.1.weight",
        "head.weight",
    ]
    assert mod.freeze_by_prefix(names, ("encoder.",)) == {
        "head.weight"
    }
    assert mod.unfreeze_by_prefix(names, ("head.",)) == {
        "head.weight"
    }


def test_discriminative_lr_groups() -> None:
    groups = mod.discriminative_lr_groups(
        ["encoder.0.weight", "head.weight"],
        rules=[("encoder.", 1e-5), ("head.", 1e-3)],
        default_lr=1e-4,
    )
    assert groups["encoder.0.weight"] == 1e-5
    assert groups["head.weight"] == 1e-3


def test_early_stopping() -> None:
    state = mod.EarlyStoppingState()
    state = mod.early_stopping_update(
        state,
        1.0,
        patience=1,
    )
    assert not state.should_stop

    state = mod.early_stopping_update(
        state,
        1.1,
        patience=1,
    )
    assert not state.should_stop

    state = mod.early_stopping_update(
        state,
        1.2,
        patience=1,
    )
    assert state.should_stop


def test_select_best_checkpoint() -> None:
    assert mod.select_best_checkpoint(
        [0.8, 0.5, 0.6],
        mode="min",
    ) == 1
    assert mod.select_best_checkpoint(
        [0.8, 0.5, 0.9],
        mode="max",
    ) == 2
