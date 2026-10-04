"""Small TensorFlow/Keras utilities used by Chapter 23."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


def visible_gpus() -> list[str]:
    return [device.name for device in tf.config.list_physical_devices("GPU")]


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
) -> keras.Model:
    if min(input_dim, hidden_dim, num_classes) <= 0:
        raise ValueError("dimensions must be positive")

    return keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(hidden_dim, activation="relu"),
            keras.layers.Dense(num_classes),
        ],
        name="tiny_classifier",
    )


def compile_classifier(
    model: keras.Model,
    *,
    learning_rate: float = 1e-3,
) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_step(
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    loss_fn: keras.losses.Loss,
    x: tf.Tensor,
    y: tf.Tensor,
) -> float:
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return float(loss.numpy())


def predict_classes(model: keras.Model, x: tf.Tensor) -> tf.Tensor:
    logits = model(x, training=False)
    return tf.argmax(logits, axis=1, output_type=tf.int64)
