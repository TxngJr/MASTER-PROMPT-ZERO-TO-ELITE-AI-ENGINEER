# Chapter 43 — Recommender Systems

## 1. What Is a Recommender?

A recommender ranks candidate items for a user/context.

~~~text
user/context
↓
candidate generation
↓
scoring/ranking
↓
top-K items
~~~

Recommendation is usually a ranking problem, not ordinary classification.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- distinguish explicit and implicit feedback
- build a user-item interaction matrix
- explain popularity baselines
- explain user/item collaborative filtering
- derive matrix factorization
- train latent factors with SGD
- explain implicit-feedback weighting
- understand BPR pairwise ranking
- explain two-tower retrieval
- distinguish retrieval from ranking
- calculate Precision@K, Recall@K and NDCG@K
- handle cold start and leakage
- design offline evaluation correctly

## 3. Explicit vs Implicit Feedback

Explicit:
- rating
- like/dislike
- star score

Implicit:
- click
- view
- watch time
- add-to-cart
- purchase

Missing implicit interaction does **not** necessarily mean dislike.

## 4. User-Item Matrix

~~~text
R ∈ R^(U×I)
~~~

Rows:
- users

Columns:
- items

Entries:
- rating / interaction strength / binary event

Real matrices are usually very sparse.

## 5. Popularity Baseline

Recommend globally popular items.

Why keep it?
- simple
- strong baseline
- useful for cold start
- reveals whether complex models add value

## 6. Collaborative Filtering

Use behavior similarity.

User-based:
- similar users → recommend what peers liked

Item-based:
- similar interaction profiles → recommend related items

## 7. Cosine Similarity

For interaction/profile vectors:

~~~text
cos(a,b)
=
a·b / (||a|| ||b||)
~~~

With L2-normalized vectors, cosine similarity equals dot product. 

## 8. Matrix Factorization

Approximate:

~~~text
R_ui
≈
p_u^T q_i
~~~

where:
- p_u ∈ R^d user latent vector
- q_i ∈ R^d item latent vector

## 9. Explicit Rating Objective

For observed ratings Ω:

~~~text
L =
Σ_(u,i)∈Ω
(R_ui - p_u^T q_i)^2
+
lambda(
||p_u||^2 + ||q_i||^2
)
~~~

Only observed ratings participate unless the problem defines missing entries otherwise.

## 10. Bias Terms

A richer model:

~~~text
r_hat_ui =
mu
+
b_u
+
b_i
+
p_u^T q_i
~~~

captures global/user/item tendencies separately from interaction factors.

## 11. Implicit Feedback

For clicks/purchases:
- positives are observed
- negatives are uncertain

Common approaches:
- sampled negatives
- confidence-weighted matrix factorization
- pairwise ranking
- retrieval objectives

## 12. Negative Sampling

Training may sample unobserved items as negatives.

Caveat:
- some unobserved items are actually unknown positives
- sampling distribution changes optimization

Document the strategy.

## 13. BPR

Bayesian Personalized Ranking compares a positive item i with negative item j:

~~~text
x_uij =
score(u,i)
-
score(u,j)

L =
-log sigma(x_uij)
~~~

Goal:
- rank observed positive above sampled negative

## 14. Two-Tower Retrieval

~~~text
user/context
↓
user tower
↓
u embedding

item features
↓
item tower
↓
v embedding

score = u·v
~~~

Item embeddings can be precomputed and indexed for ANN retrieval.

## 15. Retrieval vs Ranking

Retrieval:
- millions of items → hundreds/thousands
- fast approximate scoring

Ranking:
- smaller candidate set
- richer features
- more expensive model

Production systems often use both stages.

## 16. Ranking Features

Examples:
- user embedding
- item embedding
- recency
- price
- category
- context/time
- historical aggregates

Feature leakage is a major risk.

## 17. Temporal Split

Recommendation data is time-dependent.

Safer evaluation often uses:

~~~text
past → train
later → validation
future → test
~~~

Randomly splitting interactions can leak future behavior.

## 18. Leave-One-Out Evaluation

For each user:
- earlier interactions train
- later interaction test

Useful educational setup, but sampled-negative evaluation can inflate metrics compared with full-catalog ranking.

## 19. Precision@K

~~~text
Precision@K =
relevant items in top K / K
~~~

## 20. Recall@K

~~~text
Recall@K =
relevant items in top K
/
number of relevant items
~~~

## 21. DCG / NDCG

Discount high-rank errors more strongly.

~~~text
DCG@K =
Σ_(r=1)^K
rel_r / log2(r+1)
~~~

Then:

~~~text
NDCG@K =
DCG@K / IDCG@K
~~~

## 22. Cold Start

New user:
- no interaction history

New item:
- no collaborative signal

Solutions:
- popularity
- onboarding/preferences
- content features
- two-tower metadata
- exploration

## 23. Exploration

Pure exploitation repeatedly recommends known winners.

This can create feedback loops.

Bandits/RL may balance:
- known reward
- information gathering

But offline evaluation of exploration is difficult.

## 24. Exposure Bias

Logged data reflects what previous system showed.

No interaction may mean:
- user disliked item
- user never saw item

Logged-policy bias makes causal evaluation harder.

## 25. Diversity / Novelty

Accuracy-only ranking may produce repetitive lists.

Additional objectives:
- category diversity
- novelty
- coverage
- fairness
- business constraints

## 26. From Scratch

src/recommender_numpy.py includes:

- user_item_matrix
- popularity_scores
- cosine_similarity_matrix
- matrix_factorization_sgd
- precision_at_k
- recall_at_k
- ndcg_at_k
- bpr_pair_loss

## 27. Common Mistakes

1. treating missing implicit events as definite negatives
2. random split leaking future interactions
3. test items included in negative sampling during training
4. evaluating only hit rate with tiny sampled candidate sets
5. recommending already-consumed items unintentionally
6. no popularity baseline
7. matrix-factorization IDs out of range
8. cold-start users silently mapped to trained IDs
9. optimizing RMSE when product goal is ranking
10. reporting offline gains as guaranteed online gains

## 28. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 29. Checklist

- [ ] explicit vs implicit
- [ ] user-item matrix
- [ ] popularity baseline
- [ ] collaborative filtering
- [ ] matrix factorization
- [ ] BPR
- [ ] two-tower
- [ ] retrieval vs ranking
- [ ] Precision/Recall/NDCG@K
- [ ] cold start
- [ ] temporal leakage
- [ ] feedback loops

## 30. What's Next

Chapter 44 moves into computer-vision localization: bounding boxes, detection and segmentation.
