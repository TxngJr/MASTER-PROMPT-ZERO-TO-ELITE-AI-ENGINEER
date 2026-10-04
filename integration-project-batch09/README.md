# Batch 09 Integration Project — Sequence & Latent Representation Lab

Batch 09 has two complementary experiments.

## Part A — Sequence Memory Benchmark

Train equivalent PyTorch classifiers using:

- vanilla RNN
- LSTM
- GRU

on a synthetic delayed-memory task.

The label depends on information near the beginning of the sequence, while distracting noise continues afterward.

This exposes long-range credit-assignment behavior.

## Part B — Variational Autoencoder

Train a small VAE on sklearn digits.

Track:

- reconstruction loss
- KL divergence
- total ELBO-style loss
- latent mean/std
- prior samples

No external dataset download is required.

## Run Sequence Benchmark

~~~bash
python integration-project-batch09/src/sequence_latent_lab.py \
  --mode sequence \
  --epochs 5 \
  --output-dir reports/batch09-sequence
~~~

## Run VAE Benchmark

~~~bash
python integration-project-batch09/src/sequence_latent_lab.py \
  --mode vae \
  --epochs 5 \
  --output-dir reports/batch09-vae
~~~

## Methodology Rules

### Sequence branch

- same train/validation/test split for RNN/LSTM/GRU
- same hidden size
- same optimizer family
- same epoch budget
- validation selects model
- test evaluated once

### VAE branch

- training split only for optimization
- validation tracks reconstruction/KL
- test remains held out
- do not select latent plots by test appearance

## What To Compare

### RNN/LSTM/GRU

- parameter count
- best validation accuracy
- final test accuracy
- training time
- gradient clipping
- sequence-length sensitivity

### VAE

- reconstruction
- KL
- latent statistics
- prior-sample range
- beta sensitivity

## Required Extensions

1. sequence-length sweep
2. bidirectional offline benchmark
3. packed variable-length sequences
4. gradient norm logging
5. truncated BPTT
6. deterministic AE baseline
7. beta-VAE sweep
8. 2D latent visualization
9. latent linear-probe classifier
10. posterior-collapse diagnostics

## Mastery Questions

- Why can LSTM/GRU retain useful gradients longer than vanilla RNN?
- Why is bidirectional recurrence invalid for causal streaming?
- Why is a plain autoencoder not automatically sampleable from N(0,I)?
- Why does reparameterization preserve gradient flow?
- What does KL≈0 mean in a VAE?
