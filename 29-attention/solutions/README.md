# Chapter 29 Solutions — Key Ideas

- scores are query-key similarities.
- scaling by sqrt(d_k) controls score magnitude growth.
- softmax normalizes each query row over valid keys.
- causal masking prevents access to future positions.
- multi-head attention performs attention in several learned projection subspaces and then mixes them.
- dense self-attention creates a T×T interaction matrix.
