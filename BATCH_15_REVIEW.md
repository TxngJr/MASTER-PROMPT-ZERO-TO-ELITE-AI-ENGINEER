# Batch 15 Review — Chapters 43–45

## Chapters

- 43 — Recommender Systems
- 44 — Object Detection / Segmentation
- 45 — Embeddings / Vector Databases

## Recommender Skills

- explicit vs implicit feedback
- user-item matrices
- popularity baselines
- collaborative filtering
- matrix factorization
- user/item biases
- BPR pairwise ranking
- two-tower retrieval
- retrieval vs ranking
- Precision@K / Recall@K / NDCG@K
- temporal leakage
- cold start
- exposure bias and feedback loops

## Detection / Segmentation Skills

- xyxy / xywh / cxcywh
- box area and IoU
- pairwise IoU
- NMS
- class-aware NMS concepts
- anchors / anchor-free
- one-stage / two-stage detection
- Faster R-CNN / YOLO concepts
- semantic / instance / panoptic segmentation
- pixel cross-entropy
- mask IoU / Dice
- U-Net / Mask R-CNN
- AP / mAP concepts

## Vector Retrieval Skills

- embedding geometry
- cosine / dot / L2
- exact top-K
- ANN recall
- IVF
- HNSW
- Product Quantization
- metadata filtering
- hybrid dense+sparse retrieval
- reranking
- index lifecycle
- embedding versioning
- chunking preview for RAG

## Implemented From Scratch

### Chapter 43
- user_item_matrix
- popularity_scores
- cosine_similarity_matrix
- matrix_factorization_sgd
- precision_at_k
- recall_at_k
- ndcg_at_k
- bpr_pair_loss

### Chapter 44
- box_area
- box_iou
- pairwise_iou
- nms
- mask_iou
- dice_score

### Chapter 45
- l2_normalize
- cosine_scores
- exact_top_k
- kmeans
- IVFIndex
- ann_recall_at_k
- metadata_filter

## Integration Project

Three modes:

1. two-tower recommender with BPR-style loss
2. tiny U-Net segmentation on synthetic rectangles
3. exact-vs-IVF vector retrieval benchmark

## API / Algorithm Audit

Detection:
- box conventions are explicit
- NMS sorts by descending confidence
- lower-score boxes are suppressed when IoU exceeds threshold
- no torchvision dependency is required for the educational implementation

Vector retrieval:
- exact cosine search is used as ANN quality baseline
- IVF recall is measured against exact top-K
- nprobe controls recall/latency trade-off conceptually

## Methodology Audit

### Recommender
- sampled negatives are distinct from observed positives
- ranking metrics are used instead of classification accuracy
- cold-start and temporal leakage are documented

### Vision
- held-out segmentation samples are separate
- raw logits go to BCE-with-logits
- IoU is measured after thresholding predictions

### Vector Search
- synthetic clustered vectors make coarse search behavior visible
- nprobe=1 and nprobe=4 are compared against exact search
- ANN recall is not confused with downstream semantic quality

## Interpretation Audit

- offline recommender gains do not guarantee online lift
- mAP must specify the evaluation protocol
- NMS is post-processing, not a substitute for detector quality
- ANN search is approximate by design
- vector similarity does not prove factual/semantic equivalence
- embedding model/version changes require index governance

## Exit Gate

Before Chapter 46:

1. Batch 15 Core CI passes
2. Batch 15 PyTorch smoke passes
3. derive matrix-factorization scoring and BPR intuition
4. compute ranking metrics by hand
5. compute IoU and execute NMS by hand
6. explain one-stage vs two-stage detection
7. explain semantic vs instance segmentation
8. implement exact vector search
9. explain IVF / HNSW / PQ trade-offs
10. measure ANN Recall@K against exact retrieval
