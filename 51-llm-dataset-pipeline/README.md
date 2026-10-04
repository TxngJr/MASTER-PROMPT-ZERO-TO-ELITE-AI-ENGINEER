# Chapter 51 — LLM Dataset & Data Pipeline

## 1. Why Data Engineering Matters

A strong architecture trained on a broken corpus still produces a broken model.

LLM pretraining begins with a reproducible data system:

~~~text
raw sources
↓
ingestion
↓
normalization / cleaning
↓
filtering
↓
deduplication
↓
document-level split
↓
tokenization
↓
packing
↓
sharding
↓
streaming training
~~~

Every transformation should be versioned.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- define a document schema
- normalize text reproducibly
- clean obvious boilerplate
- hash documents
- perform exact deduplication
- explain near-duplicate detection
- split by document identity before window generation
- avoid train/validation contamination
- score/filter data quality conceptually
- handle language/domain metadata
- understand sensitive-data/PII handling requirements
- tokenize only with a frozen tokenizer
- pack token sequences efficiently
- build deterministic shards
- reason about streaming datasets
- define data mixtures
- version a dataset artifact

## 3. Document Schema

A useful record:

~~~text
{
  document_id,
  text,
  source,
  language,
  timestamp,
  license/provenance,
  metadata
}
~~~

Document identity should survive every stage.

## 4. Provenance

Record where data came from and what processing happened.

Useful fields:
- source URI/name
- acquisition date
- corpus version
- license/use constraints
- transformation version

Do not treat provenance as optional production metadata.

## 5. Unicode Normalization

Visually similar strings may use different Unicode representations.

A deterministic policy may use NFC or NFKC depending on domain.

Changing normalization after tokenizer/model training changes the effective data distribution.

## 6. Whitespace Cleaning

Common operations:
- normalize line endings
- trim trailing spaces
- collapse excessive blank lines
- preserve paragraph boundaries

Do not flatten code/Markdown where whitespace is meaningful.

## 7. Boilerplate

Examples:
- repeated navigation
- cookie banners
- headers/footers
- duplicated template text

Boilerplate can dominate token counts if repeated across many pages.

## 8. Exact Deduplication

Canonicalize then hash:

~~~text
hash(
normalized document text
)
~~~

Keep one copy for identical hashes.

This is cheap and deterministic.

## 9. Near-Duplicate Detection

Documents can differ slightly while containing nearly identical content.

Concepts:
- n-gram shingles
- Jaccard similarity
- MinHash
- locality-sensitive hashing

Near-dedup is more expensive than exact hashing.

## 10. Why Deduplicate?

Repeated content can:
- distort domain weights
- increase memorization
- waste compute
- contaminate train/eval boundaries

Deduplication policy should be measured and versioned.

## 11. Split Before Windows

Dangerous:

~~~text
document
↓ chunk/windows
↓ random train/test split
~~~

Adjacent windows from the same document can leak across splits.

Prefer:

~~~text
document
↓ document-level deterministic split
↓ tokenize/pack separately per split
~~~

## 12. Hash-Based Split

A stable hash can map document ID into:
- train
- validation
- test

Benefits:
- deterministic
- new documents do not reshuffle all old documents

## 13. Time-Based Split

For temporal evaluation:
- earlier documents → train
- later documents → validation/test

Useful when future knowledge leakage matters.

## 14. Language Detection

Track language metadata to:
- build balanced mixtures
- audit tokenizer efficiency
- evaluate multilingual coverage

Language identification is imperfect, especially for short/code-switched text.

## 15. Quality Filtering

Possible signals:
- text length
- character composition
- duplicate ratio
- boilerplate score
- language confidence
- model-based quality scores

Aggressive filtering can remove useful minority/domain data.

## 16. Safety / Sensitive Data

Data pipelines should define policy for:
- secrets/credentials
- personal information
- private records
- disallowed or restricted source material

Detection is imperfect; use layered governance and access controls.

Do not log detected sensitive values unnecessarily.

## 17. Licensing / Rights

Dataset engineering also includes:
- source permissions
- license constraints
- removal requests
- dataset lineage

Technical accessibility does not automatically imply unrestricted training rights.

## 18. Frozen Tokenizer

Once pretraining begins:

~~~text
tokenizer version = fixed
~~~

A tokenizer change modifies:
- token IDs
- vocabulary size
- sequence lengths
- embedding/LM-head compatibility

Store tokenizer version with every shard/checkpoint.

## 19. Tokenization

Document:

~~~text
text
↓ tokenizer
token IDs
~~~

Decide document separators:
- EOS between documents
- BOS/EOS protocol
- no separator only when explicitly intended

## 20. Packing

Naive padding wastes tokens.

Packing combines tokenized documents into fixed-length sequences:

~~~text
doc A tokens + EOS + doc B tokens + EOS
↓
fixed length blocks
~~~

Track boundaries if loss masking or document-aware attention is needed.

## 21. Sequence Packing

For context length T:

~~~text
packed stream
↓
blocks of T+1
↓
input block[:T]
target block[1:]
~~~

Dropping a small tail may be acceptable; document it.

## 22. Sharding

Large token corpora are split into shard files.

Benefits:
- parallel preprocessing
- streaming
- resumability
- distributed workers
- easier checksums/versioning

Shard sizes should balance file-count overhead and restart granularity.

## 23. Deterministic Shards

Given same:
- corpus version
- tokenizer
- ordering seed
- shard size

the output should be reproducible.

Store checksums.

## 24. Streaming

A streaming loader reads shards incrementally rather than materializing the full dataset in RAM.

Important state:
- shard order
- position within shard
- worker/rank assignment
- epoch/seed

Resume may need loader state for exact reproducibility.

## 25. Distributed Data Partitioning

With D workers/ranks:
- avoid accidental duplicate samples
- cover the intended corpus
- define epoch semantics

Sharding by rank and deterministic shuffle seeds are common building blocks.

## 26. Data Mixtures

Large pretraining may combine domains:

~~~text
web
code
books
science
multilingual
...
~~~

Mixture weights define how often each source contributes.

Raw corpus size alone need not determine training probability.

## 27. Mixture Sampling

Given weights w_i:

~~~text
p_i =
w_i / Σ_j w_j
~~~

Then sample source according to p_i.

Track actual tokens consumed per source, not only configured weights.

## 28. Temperature Sampling

A common balancing family transforms corpus proportions:

~~~text
p_i'
∝
p_i^alpha
~~~

alpha < 1 flattens the distribution and upweights smaller groups.

This is a design choice, not universally optimal.

## 29. Dataset Manifest

A manifest can contain:

~~~text
dataset_version
tokenizer_version
normalization_version
source counts
document counts
token counts
split policy
shard checksums
mixture weights
build config
~~~

This makes experiments auditable.

## 30. Data Validation

Before training:
- all token IDs < vocab size
- no empty shards
- split overlap = 0
- expected EOS policy
- lengths correct
- checksums match
- counts reconcile

Fail fast.

## 31. Contamination Testing

Evaluation contamination can be:
- exact
- near duplicate
- question/answer paraphrase
- benchmark included in source corpus

No one detector catches everything.

At minimum audit known eval datasets separately.

## 32. From Scratch

src/data_pipeline.py includes:

- normalize_text
- canonical_document_hash
- exact_deduplicate
- deterministic_split
- pack_token_documents
- shard_sequences
- normalize_mixture_weights
- temperature_mixture_weights

## 33. Common Mistakes

1. split after chunking
2. duplicate documents cross train/validation
3. no provenance
4. tokenizer changes without rebuilding shards
5. data cleaning destroys code formatting
6. packer loses document separators
7. shards contain invalid token IDs
8. nondeterministic corpus order without recorded seed
9. configured mixture differs from actual consumed tokens
10. no dataset manifest/checksums

## 34. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 35. Checklist

- [ ] schema / provenance
- [ ] normalization
- [ ] exact dedup
- [ ] near-dedup concepts
- [ ] split before windows
- [ ] quality filtering
- [ ] sensitive-data policy
- [ ] frozen tokenizer
- [ ] packing
- [ ] sharding
- [ ] streaming
- [ ] mixtures
- [ ] manifest
- [ ] contamination audit

## 36. What's Next

Batch 18 scales training beyond one GPU: distributed parallelism, CUDA fundamentals and mixed precision.
