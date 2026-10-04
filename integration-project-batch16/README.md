# Batch 16 Integration Project — RAG, Agent & Modern LLM Lab

Batch 16 connects retrieval, controlled tool execution and a modern decoder block.

## Part A — RAG

Internal knowledge documents are indexed with deterministic hashing embeddings.

Pipeline:

~~~text
query
├─ dense cosine retrieval
├─ sparse lexical retrieval
└─ RRF fusion
      ↓
context packing
      ↓
source IDs preserved
~~~

The lab checks that the RMSNorm source is ranked first for an RMSNorm query.

## Part B — Safe Agent

The agent has:
- read-only lookup tool
- write-side-effect tool
- strict schemas
- allow-list registry
- max-step loop

The write tool is intentionally blocked by default.

~~~text
goal
↓
lookup tool
↓
structured result
↓
finish
~~~

## Part C — Tiny Modern LLM

PyTorch decoder:

~~~text
token embedding
↓
RMSNorm
↓
RoPE + GQA + causal SDPA
↓ residual
RMSNorm
↓
SwiGLU
↓ residual
× 2 blocks
↓
final RMSNorm
↓
weight-tied LM head
~~~

Configuration:
- d_model = 32
- query heads = 4
- KV heads = 2
- 2 decoder blocks
- causal scaled-dot-product attention

The smoke path also mutates future tokens and confirms earlier logits do not change.

## Run

~~~bash
python integration-project-batch16/src/rag_agent_llm_lab.py --mode rag
python integration-project-batch16/src/rag_agent_llm_lab.py --mode agent
python integration-project-batch16/src/rag_agent_llm_lab.py --mode llm --steps 20
~~~

## Required Extensions

1. BM25-style sparse retriever
2. cross-encoder reranker
3. citation checker
4. multi-query retrieval
5. agent retry/idempotency layer
6. approval-gated write tool
7. incremental KV cache
8. cached-vs-full logit equality
9. tokenizer integration
10. pretraining dataset pipeline in Batch 17
