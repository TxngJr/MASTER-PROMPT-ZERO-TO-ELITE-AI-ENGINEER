# Chapter 46 — Retrieval-Augmented Generation (RAG)

## 1. Why RAG?

A language model's parameters are not a reliable database of every current/private fact.

RAG adds an external retrieval path:

~~~text
user question
↓
query representation
↓
retrieval
↓
relevant passages
↓
context construction
↓
LLM
↓
grounded answer
~~~

The retrieval system and the generator are separate components with separate failure modes.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- design a document-ingestion pipeline
- choose chunk size and overlap
- attach stable source metadata
- embed/index chunks
- perform dense retrieval
- perform sparse/keyword retrieval conceptually
- combine rankings with Reciprocal Rank Fusion
- rerank candidates
- construct context under a token budget
- preserve source provenance
- distinguish retrieval quality from answer quality
- calculate Recall@K / MRR for retrieval
- design grounded-answer evaluation
- detect common RAG leakage/failure modes
- understand query rewriting and multi-query retrieval

## 3. RAG Is a Pipeline

A useful decomposition:

~~~text
documents
↓
parse
↓
clean
↓
chunk
↓
metadata
↓
embed
↓
index

query
↓
rewrite/expand
↓
retrieve
↓
filter
↓
rerank
↓
context pack
↓
generate
↓
verify/cite
~~~

Debug each stage independently.

## 4. Document Parsing

Possible sources:
- plain text
- Markdown
- HTML
- PDF text
- database rows
- internal knowledge records

The parser should preserve:
- document ID
- section/page
- title
- timestamps/version
- access-control metadata

Never throw away provenance before chunking.

## 5. Cleaning

Common cleaning:
- remove navigation boilerplate
- normalize whitespace
- preserve headings
- preserve important tables/code boundaries

Over-cleaning can destroy meaning.

## 6. Chunking

Chunking determines retrieval granularity.

Too small:
- weak context
- fragmented meaning

Too large:
- less precise retrieval
- redundant tokens
- context-window waste

## 7. Fixed Token/Word Windows

Educational method:

~~~text
chunk_size = 200
overlap = 40
~~~

Then:

~~~text
chunk 0: tokens 0..199
chunk 1: tokens 160..359
...
~~~

Overlap helps preserve information crossing boundaries.

## 8. Structural Chunking

Prefer document boundaries when useful:
- heading
- paragraph
- table
- code block
- sentence

Hybrid chunking can combine semantic/structural units with size limits.

## 9. Chunk Metadata

Each chunk should carry fields such as:

~~~text
chunk_id
document_id
source
title
section
position
version
access_policy
~~~

This supports:
- citations
- filtering
- deletion
- re-indexing
- debugging

## 10. Embedding Pipeline

~~~text
chunk text
↓
embedding model
↓
vector
↓
vector index
~~~

Query must use a compatible embedding space.

Do not mix incompatible embedding model versions silently.

## 11. Dense Retrieval

Use vector similarity:

~~~text
score(q,d)
=
cos(
embed(q),
embed(d)
)
~~~

or the metric expected by the embedding model/index.

## 12. Sparse Retrieval

Keyword-oriented retrieval such as BM25 can capture:
- exact names
- rare tokens
- identifiers
- error codes
- numbers

Dense and sparse retrieval have complementary strengths.

## 13. Hybrid Retrieval

Combine ranked lists from:
- dense search
- sparse search

Then rerank/fuse.

A simple score average is often invalid because score scales differ.

## 14. Reciprocal Rank Fusion

RRF avoids raw-score calibration.

For item d:

~~~text
RRF(d)
=
Σ_r
1 / (k + rank_r(d))
~~~

where r indexes retrieval systems.

Higher fused score wins.

## 15. Candidate Retrieval vs Reranking

Stage 1:
- retrieve e.g. 50–200 candidates cheaply

Stage 2:
- rerank to top 5–20 using a stronger model

~~~text
large corpus
↓ retriever
100 candidates
↓ reranker
8 passages
↓ context
~~~

## 16. Cross-Encoder Reranking

A reranker can jointly read:

~~~text
query + candidate
~~~

and output relevance score.

Usually:
- slower
- more accurate than independent embedding similarity

Use only on a small candidate set.

## 17. Metadata Filtering

Apply:
- user permissions
- tenant
- language
- document type
- date/version

Authorization filtering must happen before unauthorized content can reach the model.

RAG must not become a permission bypass.

## 18. Query Rewriting

User question may be:
- vague
- conversational
- contain pronouns

Rewrite into retrieval-oriented query while preserving user intent.

Example concept:

~~~text
"What about its timeout?"
↓
"API gateway request timeout configuration"
~~~

Only rewrite when conversation context supports it.

## 19. Multi-Query Retrieval

Generate several retrieval formulations:

~~~text
q1
q2
q3
↓
retrieve independently
↓
merge / RRF
~~~

Can improve recall but increases cost/noise.

## 20. HyDE Concept

Hypothetical Document Embeddings:
1. generate a hypothetical answer/document
2. embed it
3. retrieve real documents near that representation

Useful in some domains, but generated hypotheses can bias retrieval.

## 21. Context Construction

Retrieved chunks must fit a finite context budget.

Possible strategy:
1. sort/rerank
2. deduplicate
3. preserve source labels
4. add until budget exhausted

Do not truncate source IDs/citation metadata accidentally.

## 22. Context Budget

If total model context is C:

~~~text
system + history + query + retrieved context + output
<= C
~~~

Reserve output tokens before filling retrieval context.

## 23. Deduplication

Overlapping chunks often repeat text.

Possible dedup signals:
- same document + nearby positions
- high lexical overlap
- high embedding similarity

Redundant context wastes tokens and can overweight one source.

## 24. Prompt Grounding Contract

A useful generation contract:

- answer from provided evidence
- distinguish evidence from inference
- cite source identifiers
- say when evidence is insufficient

Prompting alone is not a guarantee; evaluate behavior.

## 25. Citation Design

Store source IDs independently of generated prose.

A safer architecture:

~~~text
retrieved chunk
{
  text,
  source_id,
  document_id,
  location
}
~~~

The generator receives explicit source tags.

## 26. Retrieval Evaluation

For labeled relevant chunks/documents:

### Recall@K

~~~text
Recall@K =
relevant retrieved by K
/
all relevant
~~~

### MRR

~~~text
MRR =
mean(
1 / rank_first_relevant
)
~~~

## 27. Generation Evaluation

Measure separately:
- answer correctness
- faithfulness/grounding
- citation correctness
- completeness
- refusal when evidence absent
- latency/cost

A strong retriever does not guarantee a strong final answer.

## 28. RAG Failure Taxonomy

### Retrieval miss
Relevant source never retrieved.

### Ranking failure
Retrieved but ranked too low.

### Context packing failure
Relevant chunk dropped/truncated.

### Generation failure
Evidence present but model answers incorrectly.

### Citation failure
Answer correct but provenance wrong.

Classify the failure before tuning.

## 29. RAG Leakage

Common leakage:
- test question/answer in indexed training/eval corpus
- future document versions visible during historical evaluation
- unauthorized tenant documents searchable

Split and filter at document-time, not only after generation.

## 30. Advanced RAG

Concepts:
- parent-child retrieval
- sentence-window retrieval
- multi-vector retrieval
- graph RAG
- iterative retrieval
- tool-assisted retrieval
- query routing
- self-reflective retrieval

Each adds complexity; benchmark against a simple baseline.

## 31. From Scratch

src/rag_numpy.py includes:

- chunk_words
- cosine_top_k
- reciprocal_rank_fusion
- maximal_marginal_relevance
- pack_context
- retrieval_recall_at_k
- mean_reciprocal_rank

## 32. Common Mistakes

1. no exact retrieval baseline
2. chunks without stable IDs
3. mixing embedding versions
4. retrieving unauthorized documents
5. raw-score averaging across incompatible retrievers
6. no deduplication
7. context budget ignores output space
8. evaluating answer quality but not retrieval
9. citations generated without provenance mapping
10. tuning complex RAG before measuring simple RAG

## 33. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 34. Checklist

- [ ] ingestion
- [ ] chunking / overlap
- [ ] metadata / provenance
- [ ] embeddings / index
- [ ] dense / sparse
- [ ] hybrid / RRF
- [ ] reranking
- [ ] context packing
- [ ] citations
- [ ] Recall@K / MRR
- [ ] failure taxonomy
- [ ] authorization filters

## 35. What's Next

Chapter 47 turns LLMs into controlled tool-using agents with explicit state, schemas and execution boundaries.
