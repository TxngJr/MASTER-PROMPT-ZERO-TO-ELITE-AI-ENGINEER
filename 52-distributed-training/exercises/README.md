# Chapter 52 Exercises

1. Shard 103 sample IDs across 8 ranks.
2. Calculate global batch and tokens/update.
3. Average gradients from four ranks by hand.
4. Estimate ring all-reduce payload for a 2 GB gradient tensor.
5. Compare ZeRO-style stage 0–3 model-state memory.
6. Calculate pipeline bubble efficiency for multiple microbatch counts.
7. Draw data vs tensor vs pipeline parallel layouts.
8. Explain why unequal rank step counts can deadlock.
9. Design rank-aware checkpoint metadata.
10. Measure strong-scaling efficiency from benchmark timings.

Challenge:
- write a two-process CPU torch.distributed demo with Gloo
- compare DDP-equivalent averaged gradients with a single large batch
