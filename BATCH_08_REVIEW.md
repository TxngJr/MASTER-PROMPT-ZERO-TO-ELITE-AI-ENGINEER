# Batch 08 Review — Chapters 22–24

## Chapters

- 22 — PyTorch
- 23 — TensorFlow / Keras
- 24 — Convolutional Neural Networks

## PyTorch Skills

- Tensor / dtype / device
- autograd
- gradient accumulation
- nn.Module
- registered Parameters
- train() / eval()
- inference_mode()
- Dataset / DataLoader
- optimizer loop
- state_dict
- CUDA visibility and memory basics

## TensorFlow / Keras Skills

- tf.Tensor / tf.Variable
- eager execution
- GradientTape
- Sequential API
- Functional API
- Model subclassing
- compile / fit / evaluate / predict
- tf.data
- custom training loops
- .keras saving
- GPU visibility
- Keras 3 multi-backend context

## CNN Skills

- cross-correlation
- output-shape equations
- input/output channels
- parameter count
- padding / stride / dilation
- receptive field
- max pooling
- global average pooling
- NCHW vs NHWC
- PyTorch Conv2d
- Keras Conv2D

## Implemented From Scratch

- conv_output_size
- NCHW conv2d
- NCHW max pooling

## Framework Validation Strategy

Heavy frameworks are intentionally separated from core dependencies.

### Core CI

Tests:
- all previous batches
- NumPy CNN implementation
- shared integration data/split/layout helpers

### PyTorch CI

Installs CPU PyTorch and runs:
- Chapter 22 framework tests
- PyTorch CNN integration smoke test

### TensorFlow CI

Installs TensorFlow and runs:
- Chapter 23 framework tests
- TensorFlow CNN integration smoke test

GPU installation remains a local-machine concern because GitHub standard runners do not represent the Fedora RTX laptop environment.

## Methodology Audit

- same digits dataset
- same fixed stratified split
- same normalization
- equivalent small CNN topology
- validation tracked separately from final test
- framework speed claims require careful synchronization and warmup

## Common Failure Modes Covered

- device mismatch
- dtype mismatch
- target encoding mismatch
- double softmax
- gradient accumulation
- train/eval confusion
- NCHW/NHWC mismatch
- wrong convolution output math
- oversized dense classifier head

## Exit Gate

Before Chapter 25:

1. core Batch 08 CI passes
2. PyTorch smoke CI passes
3. TensorFlow smoke CI passes
4. can write a training loop in both frameworks
5. can trace every CNN tensor shape
6. can explain why framework APIs differ while the underlying optimization math is the same
