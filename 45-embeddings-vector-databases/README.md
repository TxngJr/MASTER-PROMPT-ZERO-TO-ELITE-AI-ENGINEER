# Chapter 45 — Embeddings & Vector Databases

## 1. Why Vector Search?

Embedding models map data into vectors:

~~~text
text / image / audio / item
↓
embedding model
↓
vector in R^D
~~~

Retrieval then asks:

> Which stored vectors are most similar to the query vector?

## 2. Learning Objectives

By the end of this chapter you should be able to:

- distinguish embeddings from token IDs
- explain cosine, dot-product and Euclidean distance
- implement exact nearest-neighbor search
- explain Approximate Nearest Neighbor search
- understand recall/latency trade-offs
- explain IVF
- explain HNSW conceptually
- explain Product Quantization
- understand vector normalization
- apply metadata filtering
- explain index build/update/delete concerns
- calculate Recall@K for ANN
- design a retrieval architecture
- prepare for RAG

## 3. Embedding Geometry

An embedding is a learned representation:

~~~text
x
→
e(x) ∈ R^D
~~~

Semantically related inputs should ideally occupy useful nearby regions for the task.

## 4. Similarity Metrics

### Dot Product

~~~text
score(q,x)
=
q^T x
~~~

Sensitive to vector norms.

### Cosine Similarity

~~~text
cos(q,x)
=
q^T x
/
(||q|| ||x||)
~~~

With L2-normalized vectors:

~~~text
cos(q,x)
=
q^T x
~~~

### Euclidean Distance

~~~text
d(q,x)
=
||q-x||_2
~~~

Choose the metric consistent with model training.

## 5. Exact Search

Brute-force search compares query against every vector.

Complexity for N vectors of dimension D:

~~~text
O(ND)
~~~

Pros:
- exact
- simple
- great baseline

Cons:
- expensive at large scale

## 6. Why ANN?

Approximate Nearest Neighbor indexes reduce search cost by avoiding exhaustive comparison.

Trade:

~~~text
latency / memory / index build
vs
retrieval recall
~~~

ANN is not guaranteed to return exact top-K.

## 7. Recall@K for ANN

Compare ANN result with exact top-K:

~~~text
Recall@K =
|ANN_K ∩ Exact_K|
/
K
~~~

This is an index-quality metric, not task relevance quality.

## 8. IVF — Inverted File Index

Training phase:
- cluster vectors into coarse centroids

Indexing:
- assign each vector to nearest centroid/list

Query:
1. find nearest coarse centroids
2. probe only selected lists
3. rank candidates

Important parameter:

~~~text
nprobe
~~~

Larger nprobe:
- higher recall
- higher latency

## 9. IVF Coarse Quantizer

Conceptually:

~~~text
vector space
↓ k-means
centroid 0 → list of vectors
centroid 1 → list of vectors
...
~~~

Query searches only a subset of lists.

## 10. HNSW

Hierarchical Navigable Small World is graph-based ANN.

Conceptually:
- vectors are graph nodes
- edges connect useful neighbors
- upper sparse layers support long jumps
- lower dense layer refines local search

Query greedily navigates toward closer nodes.

## 11. HNSW Parameters

Common ideas:
- M: graph connectivity
- efConstruction: build-time search effort
- efSearch: query-time search effort

Higher effort/connectivity often improves recall at memory/latency/build cost.

## 12. Product Quantization

PQ compresses vectors by splitting dimensions into subspaces:

~~~text
vector
=
[subvector_1 | subvector_2 | ... | subvector_M]
~~~

Each subvector is replaced by an index into a learned codebook.

This reduces memory and can accelerate approximate distance evaluation.

## 13. PQ Training

For each subspace:
- run k-means
- store centroid codebook
- encode each subvector by centroid ID

The result stores small integer codes rather than full float vectors.

## 14. Quantization Error

Compression introduces approximation.

Measure:
- reconstruction error
- ANN recall
- downstream retrieval quality

Higher compression generally increases error.

## 15. IVF + PQ

Common large-scale design:

~~~text
coarse IVF
+
compressed PQ codes inside lists
~~~

This reduces both search scope and storage.

## 16. Metadata Filtering

Vector similarity is often combined with structured conditions:

~~~text
language = "th"
AND
category = "manual"
AND
created_at >= ...
~~~

Filtering can happen:
- before ANN
- during ANN
- after ANN

Each has recall/performance implications.

## 17. Pre-Filter vs Post-Filter

Pre-filter:
- smaller candidate set
- may be difficult for some ANN structures

Post-filter:
- search first then remove mismatches
- can return fewer than K results

Often systems over-fetch then filter.

## 18. Hybrid Search

Combine:
- dense vector retrieval
- sparse keyword/BM25 retrieval

Then fuse/rerank.

Useful when exact keywords/numbers/entities matter.

## 19. Reranking

Stage 1:
- cheap vector ANN

Stage 2:
- more expensive cross-encoder/reranker

~~~text
millions
↓ ANN
hundreds
↓ reranker
top results
~~~

## 20. Index Lifecycle

Production vector stores need:
- insert
- update
- delete
- compaction/rebuild
- versioning
- consistency

Embedding model changes may require re-embedding the whole corpus.

## 21. Embedding Versioning

Store metadata such as:

~~~text
embedding_model
embedding_version
dimension
normalization
created_at
~~~

Never silently mix vectors from incompatible embedding spaces.

## 22. Dimensionality

Higher dimension:
- more storage
- more compute
- potentially more representational capacity

But larger is not automatically better.

## 23. Memory Estimate

Float32 storage:

~~~text
N * D * 4 bytes
~~~

Example:

~~~text
1,000,000 vectors
× 768 dims
× 4 bytes
≈ 3.07 GB
~~~

before index overhead.

## 24. Chunk Embeddings Preview

For document retrieval, long documents are often split into chunks before embedding.

Chunk design controls:
- context granularity
- number of vectors
- retrieval specificity

Chapter 46 develops this into full RAG.

## 25. Exact Search Baseline

Before tuning ANN:
1. build exact baseline
2. measure task retrieval
3. compare ANN Recall@K
4. measure latency/memory

Without exact baseline, ANN quality is difficult to diagnose.

## 26. From Scratch

src/vector_index_numpy.py includes:

- l2_normalize
- cosine_scores
- exact_top_k
- kmeans
- IVFIndex
- ann_recall_at_k
- metadata_filter

The IVF implementation is intentionally educational and small.

## 27. Common Mistakes

1. cosine search without normalization assumptions
2. mixing distance and similarity sort direction
3. ANN evaluated without exact baseline
4. too-small nprobe
5. metadata post-filter returning fewer than K
6. mixing embedding model versions
7. forgetting vector dimension validation
8. deleting metadata but leaving vectors searchable
9. assuming ANN Recall@K equals answer quality
10. choosing an index before measuring workload

## 28. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 29. Checklist

- [ ] embedding geometry
- [ ] cosine / dot / L2
- [ ] exact kNN
- [ ] ANN
- [ ] Recall@K
- [ ] IVF
- [ ] HNSW
- [ ] PQ
- [ ] metadata filters
- [ ] hybrid retrieval
- [ ] reranking
- [ ] index lifecycle

## 30. What's Next

Chapter 46 turns vector retrieval into Retrieval-Augmented Generation (RAG).
