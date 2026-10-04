# Batch 17 Integration Project — Raw Text to Tiny LLM

This batch connects the full pretraining path:

~~~text
raw documents
↓
normalize
↓
exact deduplicate
↓
document-level train/validation/test split
↓
train byte-level BPE on TRAIN only
↓
freeze tokenizer
↓
tokenize each split
↓
pack token sequences
↓
causal input/target shift
↓
Tiny Modern LLM
↓
AdamW + warmup/cosine + clipping
↓
checkpoint state_dict
~~~

## Part A — Pipeline Audit

The synthetic corpus deliberately contains:
- train documents
- validation documents
- test documents
- one normalized exact duplicate

Checks:
- duplicate removed before tokenization
- split is deterministic
- train/validation hashes have zero overlap
- train/test hashes have zero overlap
- tokenizer round-trips Thai + English text
- tokenizer merges are trained only from train text

## Part B — Tiny Pretraining

Architecture:

~~~text
byte-BPE token IDs
↓
embedding
↓
RMSNorm
↓
RoPE + GQA + causal SDPA
↓
SwiGLU
↓
final RMSNorm
↓
weight-tied LM head
~~~

Training:
- next-token cross entropy
- AdamW
- explicit warmup/cosine LR
- global norm clipping
- token counter
- state_dict checkpoint serialization smoke

## Run

~~~bash
python integration-project-batch17/src/tokenizer_dataset_pretraining_lab.py --mode pipeline

python integration-project-batch17/src/tokenizer_dataset_pretraining_lab.py \
  --mode pretraining \
  --steps 10
~~~

## Required Extensions

1. reserved BOS/EOS IDs outside base/merge IDs
2. tokenizer JSON serialization
3. near-duplicate MinHash lab
4. shard checksums/manifest files
5. resumable streaming iterator
6. gradient accumulation
7. optimizer-state checkpoint/resume
8. validation loss loop
9. greedy generation
10. cached KV decode from Batch 16
