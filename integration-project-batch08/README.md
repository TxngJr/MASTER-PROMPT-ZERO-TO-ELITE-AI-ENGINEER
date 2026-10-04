# Batch 08 Integration Project — Cross-Framework CNN Lab

This project trains equivalent small CNN classifiers on the same sklearn digits dataset using:

- PyTorch
- TensorFlow/Keras

The goal is not to declare a framework winner. The goal is to understand how the same mathematical model maps to different framework APIs.

## Shared Protocol

~~~text
sklearn digits
→ fixed stratified split
→ normalize pixels to [0,1]
→ same train / validation / test indices
→ equivalent small CNNs
→ validation tracking
→ final test
~~~

## Data Layout

PyTorch branch:

~~~text
N × C × H × W
~~~

TensorFlow/Keras branch:

~~~text
N × H × W × C
~~~

The conversion is explicit.

## Run PyTorch

Install the core course environment first:

~~~bash
python -m pip install -r requirements-batch08.txt
~~~

Then install PyTorch using the current official selector for your machine.

Run:

~~~bash
python integration-project-batch08/src/cnn_framework_lab.py \
  --framework pytorch \
  --epochs 8 \
  --output-dir reports/batch08-pytorch
~~~

## Run TensorFlow/Keras

CPU/simple install:

~~~bash
python -m pip install -r requirements-batch08-tensorflow.txt
~~~

On a compatible NVIDIA Linux GPU environment, follow the current TensorFlow Linux GPU installation instructions instead of assuming an old CUDA setup.

Run:

~~~bash
python integration-project-batch08/src/cnn_framework_lab.py \
  --framework tensorflow \
  --epochs 8 \
  --output-dir reports/batch08-tensorflow
~~~

## What To Compare

Record:

- parameter count
- best validation accuracy
- final test accuracy
- training time
- prediction time
- CPU or GPU device
- framework version

A speed comparison is meaningful only if:
- data split is identical
- model capacity is close
- batch size is equal
- preprocessing is equal
- warmup is considered
- asynchronous GPU execution is synchronized appropriately

## Required Extensions

1. confusion matrix
2. per-class recall
3. checkpoint saving
4. early stopping
5. GPU memory measurements
6. batch-size sweep
7. mixed precision later after Chapter 54 concepts
8. add augmentation
9. replace pooling with strided convolution
10. compare flatten head vs global average pooling

## Mastery Questions

- Why does PyTorch receive NCHW while Keras defaults to NHWC?
- Why should CrossEntropy-style losses receive logits?
- Why does model.eval() not itself disable PyTorch gradients?
- What does GradientTape record?
- Why can a tiny CNN be slower on GPU than CPU?
- Which parts of the training loop are framework-specific and which are mathematical?
