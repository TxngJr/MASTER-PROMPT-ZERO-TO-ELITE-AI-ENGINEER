# Chapter 62 Solutions — Key Ideas

- prefill and decode are distinct phases with different bottlenecks.
- KV-cache memory scales with layers, batch, context length, KV heads, head dimension and element width.
- paged caches replace large contiguous reservations with block mappings and reduce allocation fragmentation.
- FlashAttention is an IO-aware exact-attention implementation, not a change from quadratic dense attention arithmetic.
- continuous batching and speculation improve utilization only under suitable request/acceptance patterns.
