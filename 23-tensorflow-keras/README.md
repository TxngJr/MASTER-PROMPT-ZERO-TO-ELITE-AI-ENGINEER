# Chapter 23 — TensorFlow & Keras

## 1. Why Learn TensorFlow/Keras?

PyTorch และ TensorFlow แก้ปัญหาเดียวกันด้วย abstractions ที่ต่างกันเล็กน้อย

Mental mapping:

~~~text
PyTorch                TensorFlow / Keras
------------------------------------------------
torch.Tensor           tf.Tensor
autograd               tf.GradientTape
nn.Module              keras.Model / keras.layers.Layer
torch.optim            keras.optimizers
DataLoader             tf.data.Dataset
state_dict             model weights / .keras format
model.train/eval       training=True/False behavior
~~~

Keras 3 เป็น multi-backend API และสามารถใช้ TensorFlow, JAX หรือ PyTorch backend ได้

บทนี้ใช้ TensorFlow backend เพื่อให้เข้าใจ TensorFlow execution และ Keras training APIs

## 2. Learning Objectives

เมื่อจบบทนี้คุณควร:

- create tf.Tensor
- understand eager execution
- use GradientTape
- build Sequential models
- build Functional API models
- subclass keras.Model
- compile/fit/evaluate/predict
- write custom training step
- use tf.data
- understand training/inference mode
- save/load .keras models
- inspect GPU visibility
- debug shape/dtype/device issues

## 3. Installation

Official TensorFlow Linux docs currently provide:

~~~bash
# GPU path
python3 -m pip install 'tensorflow[and-cuda]'

# CPU path
python3 -m pip install tensorflow
~~~

For Fedora, TensorFlow officially documents Ubuntu as the supported Linux distribution, though pip installation may still work on other compatible Linux distributions.

Because your machine uses Fedora, keep TensorFlow in a dedicated virtual environment if package conflicts appear.

Verify GPU visibility:

~~~bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
~~~

## 4. Tensor Basics

~~~python
import tensorflow as tf

x = tf.constant([[1.0, 2.0]])
w = tf.Variable([[3.0], [4.0]])

y = x @ w
~~~

tf.Variable represents mutable trainable state.

## 5. Eager Execution

TensorFlow 2 executes eagerly by default:

~~~python
x = tf.constant(3.0)
print(x * x)
~~~

This feels similar to normal Python.

TensorFlow can also trace/compile computation using tf.function.

## 6. GradientTape

~~~python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x * x + 2.0 * x

grad = tape.gradient(y, x)
~~~

GradientTape records differentiable operations and performs reverse-mode differentiation.

## 7. Keras Sequential API

~~~python
model = keras.Sequential([
    keras.layers.Input(shape=(10,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3),
])
~~~

Good for simple stack-like architectures.

## 8. Functional API

Useful for:
- multiple inputs
- multiple outputs
- residual connections
- DAG architectures

~~~python
inputs = keras.Input(shape=(10,))
x = keras.layers.Dense(64, activation="relu")(inputs)
outputs = keras.layers.Dense(3)(x)
model = keras.Model(inputs, outputs)
~~~

## 9. Model Subclassing

~~~python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = keras.layers.Dense(64)
        self.fc2 = keras.layers.Dense(3)

    def call(self, x, training=False):
        x = tf.nn.relu(self.fc1(x))
        return self.fc2(x)
~~~

Closest conceptual match to PyTorch nn.Module.

## 10. compile()

~~~python
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
~~~

from_logits=True matters when final layer outputs raw logits.

Do not apply softmax twice.

## 11. fit()

~~~python
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64,
)
~~~

fit() handles:
- batching
- forward pass
- gradients
- optimizer steps
- metrics
- callbacks

## 12. evaluate() / predict()

~~~python
model.evaluate(x_test, y_test)
model.predict(x_new)
~~~

Use final test only after model/hyperparameter selection.

## 13. tf.data.Dataset

~~~python
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
~~~

Benefits:
- streaming
- batching
- preprocessing
- parallel mapping
- prefetching

## 14. Custom Training Loop

~~~python
with tf.GradientTape() as tape:
    logits = model(x, training=True)
    loss = loss_fn(y, logits)

grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
~~~

This maps directly to the tiny autodiff framework from Batch 07.

## 15. training=True / False

Layers such as:
- Dropout
- BatchNormalization

behave differently in training and inference modes.

Custom model call methods should pass training when needed.

## 16. tf.function

Decorator:

~~~python
@tf.function
def train_step(...):
    ...
~~~

TensorFlow traces Python into a graph-like executable representation.

Benefits may include:
- reduced Python overhead
- graph optimizations
- accelerator execution

But debugging traced code differs from eager debugging.

Learn eager first.

## 17. Shape Debugging

Typical image shape in TensorFlow/Keras:

~~~text
N × H × W × C
~~~

channels-last by default

PyTorch commonly uses:

~~~text
N × C × H × W
~~~

This difference causes many CNN bugs.

## 18. dtype

Typical:
- float32 for model inputs
- int32/int64 class indices
- bool masks

Inspect:

~~~python
print(tensor.shape)
print(tensor.dtype)
~~~

## 19. Saving

Keras native format:

~~~python
model.save("model.keras")
~~~

Load:

~~~python
model = keras.models.load_model("model.keras")
~~~

Save preprocessing/config/label mappings separately when required.

## 20. Callbacks

Important:
- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau
- TensorBoard

Example:

~~~python
keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)
~~~

## 21. GPU Memory

TensorFlow may allocate GPU memory differently from PyTorch.

Useful setup:

~~~python
gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
~~~

Call configuration before GPU initialization.

## 22. Keras 3 Multi-Backend Context

Standalone Keras 3 supports:
- TensorFlow
- JAX
- PyTorch

This course uses TensorFlow backend here because Chapter 22 already covers native PyTorch.

The important skill is separating:
- Keras model API
from
- backend execution system

## 23. Source

src/keras_basics.py includes:
- build_mlp
- compile_classifier
- custom GradientTape train step
- device visibility helper

## 24. Common Mistakes

1. softmax before from_logits=True loss
2. wrong target dtype
3. NCHW data fed to channels-last model
4. training=False forgotten for custom inference behavior
5. fitting preprocessing on validation/test
6. calling GPU config after runtime initialization
7. tracing everything with tf.function before code works eagerly
8. saving model but not preprocessing
9. comparing PyTorch and TensorFlow with different splits
10. assuming Keras and TensorFlow are exactly the same abstraction layer

## 25. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 26. Checklist

- [ ] tf.Tensor / tf.Variable
- [ ] GradientTape
- [ ] Sequential
- [ ] Functional API
- [ ] subclassing
- [ ] compile/fit
- [ ] custom train step
- [ ] tf.data
- [ ] saving
- [ ] GPU visibility

## 27. What's Next

Chapter 24 applies both frameworks to spatial data using convolutional neural networks.
