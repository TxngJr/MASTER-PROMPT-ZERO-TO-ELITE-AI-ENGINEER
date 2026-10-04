# Batch 17 Review — Chapters 49–51

## Chapters

- 49 — Tokenizer From Scratch
- 50 — LLM Pretraining From Scratch
- 51 — LLM Dataset / Data Pipeline

## Tokenizer Skills

- Unicode vs UTF-8 bytes
- byte-level base vocabulary
- BPE pair counting
- deterministic merge tie-breaking
- merge ranks
- encode / decode
- special-token design
- vocabulary-size trade-offs
- bytes-per-token evaluation
- multilingual efficiency
- tokenizer artifact/versioning

## Pretraining Skills

- causal input/target shifting
- cross-entropy / perplexity
- fixed-length token windows
- AdamW
- warmup
- cosine decay
- gradient accumulation
- gradient clipping
- validation loss
- tokens seen / throughput
- checkpoint/resume state
- first-100-step debugging

## Data-Pipeline Skills

- document schema / provenance
- Unicode normalization
- boilerplate/quality filtering concepts
- exact deduplication
- near-dedup concepts
- document-level deterministic split
- frozen tokenizer
- token packing
- sharding
- streaming-state concepts
- data mixtures
- manifest/versioning
- contamination audits

## Implemented From Scratch

### Chapter 49
- bytes_to_base_tokens
- pair_counts
- merge_pair
- train_bpe
- ByteBPETokenizer
- encode / decode
- bytes_per_token

### Chapter 50
- make_causal_windows
- stable_cross_entropy
- perplexity_from_loss
- warmup_cosine_lr
- effective_tokens_per_update
- global_norm
- clip_by_global_norm

### Chapter 51
- normalize_text
- canonical_document_hash
- exact_deduplicate
- deterministic_split
- pack_token_documents
- shard_sequences
- normalize_mixture_weights
- temperature_mixture_weights

## Integration Project

Pipeline:

~~~text
raw documents
↓
normalize
↓
exact deduplicate
↓
document-level split
↓
train BPE on TRAIN only
↓
freeze tokenizer
↓
tokenize all splits
↓
pack T+1 sequences
↓
causal shift
↓
Tiny Modern LLM
↓
AdamW + warmup/cosine + clipping
↓
state_dict checkpoint smoke
~~~

## Data-Leakage Audit

The integration verifies:
- exact normalized duplicate removed
- train/validation document hashes do not overlap
- train/test document hashes do not overlap
- tokenizer merge training reads train text only
- validation/test documents are tokenized only after tokenizer is frozen

## Tokenizer Audit

- base alphabet is UTF-8 bytes
- deterministic merge tie-breaking
- merge ranks preserved during encoding
- Thai + English roundtrip test
- invalid token IDs fail closed

## Pretraining Audit

- target is shifted by one token
- raw logits go directly to cross-entropy
- AdamW optimizer used in framework lab
- explicit warmup/cosine LR
- global-norm clipping
- tokens_seen tracked
- checkpoint uses state_dict serialization

Current PyTorch guidance continues to recommend saving module state dictionaries for portability.

## Interpretation Audit

- lower tokenizer token count does not automatically mean better semantics
- perplexity comparisons require the same tokenizer/evaluation protocol
- one tiny training run does not demonstrate scaling behavior
- deduplication reduces repetition but does not prove benchmark decontamination
- data filtering can introduce domain/language bias
- technical access to data does not imply unrestricted training rights

## Exit Gate

Before Chapter 52:

1. Batch 17 Core CI passes
2. Batch 17 PyTorch smoke passes
3. explain UTF-8 byte-level tokenization
4. train/encode/decode deterministic BPE
5. build shifted causal targets
6. derive warmup + cosine schedule
7. explain accumulation / clipping
8. build document-level split before windowing
9. deduplicate and pack token sequences
10. produce a dataset/tokenizer version contract
