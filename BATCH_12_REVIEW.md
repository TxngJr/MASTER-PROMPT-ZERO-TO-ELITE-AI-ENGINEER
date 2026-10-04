# Batch 12 Review — Chapters 34–36

## Chapters

- 34 — BERT / Encoder-Only Models
- 35 — GPT / Decoder-Only Models
- 36 — Encoder-Decoder / T5

## BERT Skills

- bidirectional encoder attention
- MLM
- 15% token selection
- original 80/10/10 corruption policy
- CLS / SEP / MASK / PAD
- segment embeddings
- masked-only loss
- classifier fine-tuning
- MLM vs autoregressive generation

## GPT Skills

- decoder-only causal Transformer
- shifted next-token targets
- causal masking
- autoregressive generation
- greedy decoding
- temperature
- top-k
- top-p / nucleus sampling
- context windows
- KV-cache motivation
- training vs generation compute

## T5 / Encoder-Decoder Skills

- encoder memory
- decoder causal self-attention
- cross-attention
- teacher forcing
- shift-right
- source/target padding masks
- text-to-text framing
- span corruption
- sentinel tokens
- relative-position concepts
- inference loops

## Implemented From Scratch

### Chapter 34
- add_special_tokens
- create_token_type_ids
- mlm_corrupt
- masked_accuracy

### Chapter 35
- make_causal_lm_pairs
- stable_softmax
- apply_temperature
- top_k_filter
- top_p_filter
- sample_from_logits

### Chapter 36
- shift_right
- make_padding_mask
- span_corrupt

## PyTorch Integration

The Batch 12 lab trains:

1. Tiny BERT-style masked encoder
2. Tiny GPT-style causal LM
3. Tiny T5-style encoder-decoder model

All use internal/synthetic data so CI does not depend on network dataset downloads.

## Information-Flow Audit

### BERT
- future tokens are visible
- only selected MLM positions contribute to loss
- special tokens are excluded from corruption logic in the chapter helper

### GPT
- future tokens are hidden by causal mask
- labels are shifted by one
- generation consumes model outputs sequentially

### T5
- encoder sees full source
- decoder sees only previous target positions
- cross-attention reads encoder memory
- decoder inputs are shifted right

## Interpretation Audit

- BERT MLM is not a standard causal generator
- GPT next-token loss is not the same objective as MLM
- T5 is not simply "BERT plus a decoder"; its seq2seq information flow and span-corruption objective differ
- tiny synthetic CI models validate mechanics, not real pretrained-model quality

## Exit Gate

Before Chapter 37:

1. Batch 12 Core CI passes
2. Batch 12 PyTorch smoke passes
3. can explain BERT/GPT/T5 information flow without notes
4. can build MLM corruption labels
5. can implement top-k/top-p filtering
6. can explain KV-cache motivation
7. can construct shifted decoder targets
8. can construct T5 span-corruption source/target pairs
