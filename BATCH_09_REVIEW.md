# Batch 09 Review — Chapters 25–27

## Chapters

- 25 — Recurrent Neural Networks
- 26 — LSTM / GRU
- 27 — Autoencoders / Variational Autoencoders

## Sequence Skills

- hidden state
- recurrent parameter sharing
- unrolling
- BPTT
- vanishing/exploding gradients
- gradient clipping
- truncated BPTT
- variable-length sequences
- packed sequences
- masking
- bidirectional caveats

## LSTM / GRU Skills

- LSTM forget/input/candidate/output gates
- cell state
- GRU reset/update/candidate gates
- parameter-count reasoning
- multi-layer state shapes
- bidirectional state shapes
- padded-vs-valid final state
- gate saturation diagnostics

## Representation-Learning Skills

- encoder / decoder
- bottleneck
- denoising autoencoder
- deterministic AE limitations
- VAE approximate posterior
- mu / logvar
- reparameterization
- diagonal Gaussian KL
- ELBO
- beta-VAE
- posterior collapse
- prior sampling

## Implemented From Scratch

### Chapter 25
- TanhRNN step
- sequence forward pass
- sequence readout

### Chapter 26
- stable sigmoid
- LSTMCellNumPy
- GRUCellNumPy
- recurrent sequence forwards

### Chapter 27
- reparameterization
- Gaussian KL to N(0,I)
- stable BCE with logits
- VAE loss decomposition

## Framework Integration

PyTorch integration includes:

- nn.RNN
- nn.LSTM
- nn.GRU
- gradient clipping
- validation checkpointing
- tiny VAE
- BCE-with-logits reconstruction
- KL regularization
- prior sampling

## Methodology Audit

### Sequence branch

- same data split for RNN/LSTM/GRU
- same hidden size
- same optimization budget
- validation selects model
- final test held out

### VAE branch

- training optimization separated from test evaluation
- reconstruction and KL tracked separately
- latent prior sampling uses N(0,I)
- no claim that low-dimensional plots prove disentanglement

## Important Caveats

- batch_first does not change recurrent hidden-state layer/direction layout
- bidirectional models use future context
- padded final timestep can be invalid
- PyTorch GRU candidate equation differs subtly from some textbook/original-paper forms
- plain autoencoder latent space is not automatically a valid simple generative prior
- KL near zero may indicate posterior collapse

## Exit Gate

Before Chapter 28:

1. core Batch 09 tests pass
2. PyTorch Batch 09 smoke tests pass
3. can derive vanilla RNN recurrence
4. can explain repeated-Jacobian gradient problems
5. can write LSTM and GRU equations from memory
6. can explain reparameterization and ELBO
7. can diagnose posterior collapse conceptually
8. can run both Batch 09 integration modes
