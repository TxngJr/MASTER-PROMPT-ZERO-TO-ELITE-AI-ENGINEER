from pathlib import Path
import importlib.util

import pytest

torch = pytest.importorskip("torch")


MODULE_PATH = Path(__file__).parents[1] / "src" / "pytorch_basics.py"
SPEC = importlib.util.spec_from_file_location("pytorch_basics_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_cpu_forward_shape_and_parameter_count() -> None:
    mod.seed_everything(7)
    model = mod.TinyClassifier(4, 8, 3)
    x = torch.randn(5, 4)
    logits = model(x)

    assert logits.shape == (5, 3)
    assert mod.parameter_count(model) == (4 * 8 + 8) + (8 * 3 + 3)


def test_train_step_changes_parameters() -> None:
    mod.seed_everything(9)
    model = mod.TinyClassifier(2, 8, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 1, 1], dtype=torch.long)

    before = [p.detach().clone() for p in model.parameters()]
    loss = mod.train_step(model, optimizer, criterion, x, y)

    assert loss > 0.0
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, model.parameters())
    )


def test_predict_classes_disables_training_mode() -> None:
    model = mod.TinyClassifier(3, 5, 2)
    x = torch.randn(4, 3)
    pred = mod.predict_classes(model, x)

    assert pred.shape == (4,)
    assert model.training is False
