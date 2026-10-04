from pathlib import Path
import importlib.util

import pytest

tf = pytest.importorskip("tensorflow")


MODULE_PATH = Path(__file__).parents[1] / "src" / "keras_basics.py"
SPEC = importlib.util.spec_from_file_location("keras_basics_course", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_mlp_forward_shape_and_parameter_count() -> None:
    tf.keras.utils.set_random_seed(7)
    model = mod.build_mlp(4, 8, 3)
    x = tf.ones((5, 4), dtype=tf.float32)
    logits = model(x, training=False)

    assert tuple(logits.shape) == (5, 3)
    assert model.count_params() == (4 * 8 + 8) + (8 * 3 + 3)


def test_custom_train_step_changes_weights() -> None:
    tf.keras.utils.set_random_seed(9)
    model = mod.build_mlp(2, 8, 2)

    x = tf.constant(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=tf.float32,
    )
    y = tf.constant([0, 1, 1, 1], dtype=tf.int64)

    _ = model(x)
    before = [w.numpy().copy() for w in model.trainable_variables]

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = mod.train_step(model, optimizer, loss_fn, x, y)

    assert loss > 0.0
    assert any(
        not (a == b.numpy()).all()
        for a, b in zip(before, model.trainable_variables)
    )


def test_predict_classes_shape() -> None:
    model = mod.build_mlp(3, 5, 2)
    x = tf.ones((4, 3), dtype=tf.float32)
    pred = mod.predict_classes(model, x)
    assert tuple(pred.shape) == (4,)
