# Batch 15 Integration Project — Recommendation, Vision & Vector Retrieval Lab

Batch 15 contains three experiments.

## Part A — Two-Tower Recommender

Synthetic users/items are generated from hidden preference factors.

~~~text
user features -> user tower
                          \
                           dot-product affinity
                          /
item features -> item tower
~~~

Training uses sampled pairwise BPR-style loss.

Tracks:
- training loss
- parameter count
- Recall@10

## Part B — Tiny U-Net Segmentation

Synthetic noisy rectangle images:

~~~text
image
↓
encoder
↓
bottleneck
↓ upsample + skip
↓
binary mask logits
~~~

Tracks:
- BCE-with-logits loss
- parameter count
- held-out mask IoU

## Part C — Exact vs IVF Vector Search

Synthetic clustered embeddings:

~~~text
query
├─ exact cosine top-K
└─ IVF top-K
~~~

Compare:
- nprobe=1
- nprobe=4
- exact top-K

Track ANN Recall@10.

## Run

~~~bash
python integration-project-batch15/src/recommendation_vision_retrieval_lab.py --mode recommender --steps 80
python integration-project-batch15/src/recommendation_vision_retrieval_lab.py --mode segmentation --steps 40
python integration-project-batch15/src/recommendation_vision_retrieval_lab.py --mode vector-search
~~~

## Required Extensions

1. temporal recommender split
2. two-stage retrieval/ranking
3. recommendation cold-start features
4. class-aware detection NMS
5. synthetic box detector
6. multiclass segmentation
7. IVF nprobe latency benchmark
8. Product Quantization
9. metadata-filter over-fetch
10. hybrid dense+sparse retrieval
