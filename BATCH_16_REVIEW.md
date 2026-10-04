# Batch 16 Review — Chapters 46–48

## Chapters

- 46 — Retrieval-Augmented Generation
- 47 — AI Agents / Tool Calling
- 48 — Modern LLM Architecture

## RAG Skills

- parsing / cleaning / chunking
- overlap and structural chunking
- provenance metadata
- dense retrieval
- sparse retrieval
- hybrid retrieval
- Reciprocal Rank Fusion
- reranking
- MMR diversification
- context packing
- citation/source tracking
- Recall@K / MRR
- failure taxonomy
- authorization-aware retrieval

## Agent Skills

- explicit agent loop
- model vs executor separation
- structured tool schemas
- argument validation
- allow-list registry
- state management
- step/tool budgets
- planning vs execution
- retries / idempotency
- read vs write permissions
- prompt/tool-output injection boundaries
- approval gates
- observability / evaluation

## LLM Architecture Skills

- token embeddings
- residual stream
- pre-norm
- RMSNorm
- RoPE
- scaled dot-product attention
- causal masking
- MHA / MQA / GQA
- SwiGLU
- final normalization
- weight-tied LM head
- KV-cache shapes/memory
- prefill vs decode
- attention implementation considerations

## Implemented From Scratch

### Chapter 46
- chunk_words
- cosine_top_k
- reciprocal_rank_fusion
- maximal_marginal_relevance
- pack_context
- retrieval_recall_at_k
- mean_reciprocal_rank

### Chapter 47
- ToolSpec
- validate_arguments
- ToolRegistry
- ToolResult
- AgentState
- bounded_agent_loop

### Chapter 48
- rms_norm
- rope_angles
- apply_rope
- repeat_kv
- causal_attention
- swiglu
- kv_cache_bytes

## Integration Project

Three modes:

1. dense+sparse RAG with RRF/context packing
2. allow-listed read-only agent with blocked write side effect
3. tiny decoder-only LLM with RMSNorm, RoPE, GQA, SDPA and SwiGLU

## PyTorch API Audit

Current PyTorch provides:
- nn.RMSNorm
- scaled_dot_product_attention
- optimized SDPA backends depending on platform

The Batch 16 LLM lab uses the framework SDPA primitive while preserving the architecture math in the chapter's NumPy implementation.

## Methodology Audit

### RAG
- source IDs survive retrieval/context packing
- dense and sparse rankings are fused rather than raw-score averaged
- retrieval and generation concerns remain separate
- authorization is documented as pre-generation filtering

### Agent
- arbitrary tool names fail closed
- arguments are schema validated
- write tools are blocked by default
- step budget prevents infinite loops
- tool outputs are observations rather than permission instructions

### LLM
- future-token mutation test checks causal isolation
- GQA uses fewer KV than query heads
- LM head is weight tied
- no softmax is applied before cross-entropy
- KV-cache memory uses H_kv, not H_q

## Interpretation Audit

- RAG does not guarantee factuality by itself
- citations require real provenance mapping
- tool calling does not imply unrestricted autonomy
- hidden reasoning is not required for reliable action traces
- attention optimization does not change causal semantics
- KV cache lowers repeated computation but consumes growing memory

## Exit Gate

Before Chapter 49:

1. Batch 16 Core CI passes
2. Batch 16 PyTorch smoke passes
3. build/evaluate a basic RAG retriever
4. explain retrieval-vs-generation failure
5. implement strict tool schemas/allow-listing
6. explain injection and least privilege
7. derive RMSNorm
8. implement RoPE
9. explain MHA/MQA/GQA
10. calculate KV-cache memory
11. trace a full modern decoder block
