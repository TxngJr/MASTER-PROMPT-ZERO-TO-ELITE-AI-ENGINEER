# Mini Project — Tiny CLIP & Fusion Lab

Create paired synthetic modalities generated from the same hidden semantic vector.

Train:
- image encoder
- text encoder
- projection heads
- symmetric contrastive objective

Evaluate:
- image→text Recall@1/5
- text→image Recall@1/5
- cosine alignment
- shuffled-pair baseline

Extension:
- add token-level cross-attention and a small classification task requiring both modalities.
