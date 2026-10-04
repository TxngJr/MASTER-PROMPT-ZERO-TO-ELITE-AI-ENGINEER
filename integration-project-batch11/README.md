# Batch 11 Integration Project — Vision & Static Embedding Lab

Batch 11 has two experiments.

## Part A — Tiny Vision Transformer

Use sklearn digits:

~~~text
8×8 image
↓
2×2 patch embedding
↓
16 patch tokens
↓
prepend CLS
↓
learned positions
↓
2 Transformer encoder layers
↓
CLS classifier
~~~

Tracks:
- parameter count
- patch size
- token count
- validation accuracy
- final test accuracy
- CPU/CUDA device

## Part B — Static Word Embeddings

Use an internal corpus and train:

- Skip-Gram with Negative Sampling
- GloVe
- FastText-style subword table

Tracks:
- loss histories
- vocabulary size
- training-pair count
- cosine similarities
- OOV subword-vector norm

## Run ViT

~~~bash
python integration-project-batch11/src/vision_nlp_lab.py \
  --mode vit \
  --epochs 8 \
  --output-dir reports/batch11-vit
~~~

## Run Embeddings

~~~bash
python integration-project-batch11/src/vision_nlp_lab.py \
  --mode embeddings \
  --epochs 8 \
  --output-dir reports/batch11-embeddings
~~~

## Methodology Rules

ViT:
- same fixed split throughout model selection
- validation chooses checkpoint
- final test once
- full self-attention; no causal mask

Embeddings:
- corpus is fixed before evaluation
- no cherry-picked single analogy as the only metric
- static-vector similarity is treated as a diagnostic

## Required Extensions

1. Tiny CNN vs ViT comparison
2. patch-size sweep
3. mean-pooling vs CLS
4. learned vs sinusoidal image positions
5. Skip-Gram subsampling
6. GloVe epoch sweep
7. FastText hashed buckets
8. nearest-neighbor reports
9. downstream linear classifier
10. seed stability report
