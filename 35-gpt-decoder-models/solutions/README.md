# Chapter 35 Solutions — Key Ideas

- GPT-style models optimize left-to-right next-token prediction.
- causal masking prevents target leakage.
- temperature rescales logits before sampling.
- top-k fixes candidate count; top-p adapts candidate count to probability mass.
- KV caching reuses past key/value projections but still stores context-dependent memory and attends over prior positions.
