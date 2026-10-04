# Chapter 72 — Long Context & Memory

## 1. Context Is Not One Thing

Long-context systems combine several different ideas:

- positional representation
- attention pattern
- KV-cache policy
- context selection/compression
- retrieval
- persistent external memory

Extending one does not automatically solve the others.

## 2. Learning Objectives

- derive dense causal attention-pair growth
- derive sliding-window attention cost
- understand RoPE and scaling concepts
- distinguish interpolation/extrapolation
- understand current RoPE variants
- calculate KV-cache growth
- understand bounded caches for sliding/chunked layers
- design chunking/context-selection policies
- distinguish context compression from retrieval
- design external memory scoring
- evaluate long-context retrieval and position sensitivity

## 3. Dense Causal Attention

For sequence length N, a causal token attends to current/past positions.

Number of allowed token pairs:

~~~text
N(N+1)/2
~~~

So pairwise attention arithmetic grows quadratically.

## 4. Sliding-Window Attention

Each token attends only to up to W recent positions.

~~~text
pairs = sum_t min(t+1, W)
~~~

For long N and fixed W, this approaches O(NW) rather than O(N²).

Trade-off: distant tokens cannot directly communicate through one layer.

## 5. Global / Sparse Patterns

Other sparse attention designs may combine:
- local windows
- selected global tokens
- blocks
- strided connections

The exact connectivity pattern determines both compute and information flow.

## 6. RoPE

Rotary Position Embeddings encode position by rotating query/key components as a function of position.

They enable relative-position structure without simply adding a learned position vector.

## 7. RoPE Scaling

Extending beyond training context often requires modifying positional frequencies/positions rather than merely increasing one config integer.

Transformers currently documents RoPE types including default, linear, dynamic/NTK, YaRN, LongRoPE and Llama3-style scaling. citeturn744269search1turn744269search6

Each variant has its own assumptions and configuration.

## 8. Linear Scaling Intuition

A simple conceptual interpolation:

~~~text
scaled_position = position / factor
~~~

This maps a longer inference range into a smaller positional-frequency range.

Real implementations operate on RoPE frequencies/configuration and should use the model-supported method.

## 9. Position Extrapolation

A model seeing mathematically valid RoPE values at a longer length does not guarantee useful language-model behavior there.

Evaluate:
- perplexity/quality
- retrieval
- position sensitivity
- generation stability

## 10. KV Cache

For ordinary autoregressive decoding:

~~~text
KV bytes
=
2 * layers * sequence_length
* kv_heads * head_dim
* bytes_per_element * batch
~~~

Long context can become cache-memory limited even when model weights fit.

## 11. Sliding / Chunked Cache Bounds

Current Transformers cache documentation notes that cache growth stops at the configured sliding-window/chunk size for layers using sliding or chunked attention. citeturn315138search6

That does not imply every layer in every model has bounded cache.

## 12. MHA / GQA / MQA

Fewer KV heads reduce long-context cache cost.

This connects directly to Chapters 48 and 62.

## 13. Context Chunking

Split documents/conversations into chunks:

~~~text
[0:chunk]
[chunk-overlap:...]
...
~~~

Overlap reduces boundary loss but duplicates tokens.

## 14. Chunk Size

Too small:
- missing cross-chunk context
- retrieval fragmentation

Too large:
- expensive embedding/reranking/context
- irrelevant tokens consume budget

Choose using retrieval/application evaluation.

## 15. Context Packing

Given a token budget, select the most useful segments rather than simply appending everything.

Possible signals:
- relevance
- recency
- importance
- source priority

## 16. Context Compression

Compression reduces the representation of already-known context:
- extractive selection
- summarization
- learned compression
- memory tokens

Compression can lose details, so keep provenance/backreferences where possible.

## 17. Retrieval Memory

Store information externally and retrieve only when relevant.

~~~text
current query
↓
retriever
↓
memory store
↓
selected memories
↓
context
~~~

This avoids requiring all historical information inside the active context window.

## 18. Conversation Memory

Possible layers:
- recent-turn buffer
- rolling summary
- durable structured facts/preferences
- episodic retrieval store

Each should have explicit retention and update semantics.

## 19. Memory Scoring

One educational score:

~~~text
score
=
w_s * similarity
+ w_r * recency
+ w_i * importance
~~~

The weights are product/application policy, not universal constants.

## 20. Memory Write Policy

Do not persist every message automatically.

Decide:
- what is durable
- who owns it
- expiry
- correction/deletion
- conflict handling
- privacy/access policy

## 21. Memory Retrieval Failure

Failures include:
- relevant memory missed
- stale memory retrieved
- conflicting memories
- too much retrieved context
- cross-user/tenant leakage

Retrieval memory inherits RAG authorization requirements from Chapters 46 and 68.

## 22. Long-Context Evaluation

Do not test only short prompts after extending context.

Evaluate across positions and lengths:
- beginning
- middle
- end
- multiple distractors
- multiple relevant facts

## 23. Needle Retrieval

A basic synthetic test places a known fact at different positions and asks the model to retrieve it.

Useful but insufficient: real workloads require reasoning and multi-document integration.

## 24. Recall@K for Memory

~~~text
Recall@K
=
relevant memories retrieved in top K
/ total relevant memories
~~~

Also evaluate whether the generator actually uses retrieved evidence correctly.

## 25. Lost-in-the-Middle Effects

Performance can depend strongly on where relevant content appears.

Measure position curves instead of reporting one aggregate score.

## 26. Long Context vs RAG

Not mutually exclusive.

Long context:
- keeps more active tokens

RAG:
- chooses external information

A practical system may retrieve into a long context window.

## 27. Long Context vs Persistent Memory

Long context is temporary model input.

Persistent memory survives across requests/sessions in external state.

Do not conflate the two.

## 28. From Scratch

`src/long_context.py` implements:
- causal_dense_attention_pairs
- sliding_window_attention_pairs
- attention_pair_reduction
- kv_cache_bytes
- bounded_sliding_cache_tokens
- linear_rope_scaled_position
- chunk_ranges
- select_context_segments
- memory_score
- retrieval_recall_at_k

## 29. Common Mistakes

1. max_position_embeddings changed without validation
2. RoPE scaling assumed to guarantee quality
3. sliding window described as full global attention
4. KV-cache memory ignored
5. every historical turn appended forever
6. chunk overlap duplicates ignored
7. summary memory treated as lossless
8. retrieval access control omitted
9. needle benchmark treated as complete long-context evaluation
10. long context confused with persistent memory

## 30. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 31. Checklist

- [ ] dense attention scaling
- [ ] sliding window
- [ ] RoPE scaling
- [ ] KV cache
- [ ] chunking
- [ ] context packing
- [ ] compression
- [ ] retrieval memory
- [ ] persistence policy
- [ ] long-context evaluation

## 32. What's Next

Batch 25 moves into reasoning models, vision-language models and modern audio/voice systems.