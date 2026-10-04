# Batch 11 Review — Chapters 31–33

## Chapters

- 31 — Vision Transformer
- 32 — NLP Fundamentals
- 33 — Word2Vec / GloVe / FastText

## Vision Transformer Skills

- patch counting
- NCHW patchification
- patch projection
- Conv2D patch embedding equivalence
- CLS token
- learned positional embeddings
- Transformer encoder
- patch-size / attention-cost trade-offs
- CNN vs ViT inductive bias

## NLP Foundation Skills

- Unicode / code points
- normalization policy
- character / word / subword levels
- vocabulary / UNK / PAD
- train-only vocabulary construction
- n-grams
- smoothing
- next-token windows
- cross-entropy / perplexity
- padding / attention / loss masks
- corpus leakage and quality

## Static Embedding Skills

- CBOW / Skip-Gram
- input vs output embedding tables
- negative sampling
- unigram^0.75 distribution
- cosine similarity
- co-occurrence matrices
- GloVe weighted objective
- character n-grams
- FastText-style OOV vectors
- intrinsic vs extrinsic evaluation
- embedding bias

## Implemented From Scratch

### Chapter 31
- patch_count
- patchify_nchw
- linear_patch_embedding
- prepend_cls_token
- add_learned_positions

### Chapter 32
- Unicode normalization
- educational tokenizer
- Vocabulary
- n-gram counting
- smoothed NGramLanguageModel
- perplexity
- next-token window construction

### Chapter 33
- Skip-Gram pair generation
- negative-sampling distribution
- SkipGramNegativeSampling
- co-occurrence matrix
- GloVeModel
- character n-grams
- FastTextSubwordTable
- cosine similarity

## Integration

The Batch 11 lab includes:

1. Tiny Vision Transformer on sklearn digits
2. Static embedding benchmark on an internal corpus

## Methodology Audit

### ViT
- fixed train/validation/test split
- validation checkpoint selection
- final test separate
- no causal mask for image classification
- patch count/position length explicit

### NLP / Embeddings
- vocabulary concepts distinguish IDs from vectors
- perplexity comparisons require compatible tokenization
- static embedding similarities are diagnostics, not semantic truth
- corpus bias is explicitly documented

## PyTorch API Audit

The educational Tiny ViT uses TransformerEncoderLayer with:
- batch_first=True
- norm_first=True
- GELU
- a small feed-forward dimension

PyTorch documents TransformerEncoderLayer as a reference implementation of the original Transformer architecture, which is appropriate for foundational teaching.

## Exit Gate

Before Chapter 34:

1. Batch 11 Core CI passes
2. Batch 11 PyTorch smoke passes
3. can patchify an image by hand
4. can trace ViT shapes end to end
5. can explain Unicode/tokenization/vocabulary separation
6. can calculate n-gram probabilities/perplexity
7. can derive Skip-Gram negative sampling intuition
8. can explain GloVe vs Word2Vec vs FastText
