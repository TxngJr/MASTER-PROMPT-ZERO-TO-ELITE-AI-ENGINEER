# Batch 10 Review — Chapters 28–30

## Chapters

- 28 — Generative Adversarial Networks
- 29 — Attention
- 30 — Transformer

## GAN Skills

- generator / discriminator
- minimax objective
- non-saturating generator objective
- stable BCE with logits
- alternating optimization
- detach semantics
- mode collapse
- fixed-noise monitoring
- DCGAN principles
- Wasserstein / gradient-penalty intuition

## Attention Skills

- Query / Key / Value
- dot-product scores
- sqrt(d_k) scaling
- stable softmax
- padding masks
- causal masks
- self-attention
- cross-attention
- multi-head split/combine
- attention complexity
- PyTorch scaled_dot_product_attention

## Transformer Skills

- token embeddings
- positional information
- sinusoidal positions
- residual connections
- LayerNorm
- position-wise FFN
- encoder blocks
- decoder blocks
- pre-norm / post-norm
- decoder-only causal Transformers
- vocabulary projection
- next-token cross-entropy
- autoregressive generation

## Implemented From Scratch

### Chapter 28
- stable softplus
- BCE with logits
- discriminator loss
- non-saturating generator loss
- saturating minimax generator objective

### Chapter 29
- stable softmax
- causal mask
- scaled dot-product attention
- split/combine heads
- multi-head self-attention

### Chapter 30
- sinusoidal position encoding
- LayerNorm
- GELU
- causal pre-norm Transformer block
- residual/FFN mechanics

## PyTorch Integration

The Batch 10 integration lab trains:

1. a 2D MLP GAN
2. a tiny decoder-only character Transformer language model

The attention test also compares the NumPy implementation directly against PyTorch SDPA on the same tensors.

## Methodology Audit

### GAN

- fake samples detached only during D update
- G update keeps gradient path through D to G
- fixed latent probe
- real/fake distributions compared, not only cherry-picked samples

### Transformer

- corpus split before training windows
- causal mask
- raw logits into cross-entropy
- validation checkpoint selection
- generation after training

## Performance Perspective

Dense attention creates T×T interactions.

Modern PyTorch can dispatch scaled dot-product attention to optimized fused implementations depending on hardware and inputs.

The mathematical model remains the same even when the kernel implementation changes.

## Interpretation Audit

- GAN loss is not a direct image/sample-quality score
- attention maps are not automatically causal explanations
- a Transformer is more than attention
- generated text quality from a tiny toy corpus is not evidence of general language understanding

## Exit Gate

Before Chapter 31:

1. all Batch 10 core tests pass
2. Batch 10 PyTorch smoke tests pass
3. derive GAN objectives
4. implement scaled attention from memory
5. explain causal vs padding masks
6. trace every multi-head shape
7. assemble a Transformer block
8. explain next-token training and generation
