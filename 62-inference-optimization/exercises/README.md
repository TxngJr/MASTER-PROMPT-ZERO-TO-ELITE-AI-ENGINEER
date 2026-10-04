# Chapter 62 Exercises

1. Calculate KV-cache memory for MHA/GQA/MQA.
2. Compare DynamicCache and StaticCache trade-offs.
3. Calculate padding efficiency for variable-length batches.
4. Calculate paged-block tail fragmentation.
5. Estimate prefix-cache prefill savings.
6. Explain why decode can be memory-bandwidth bound.
7. Explain FlashAttention without claiming linear dense attention.
8. Measure speculative acceptance rate.
9. Compare TTFT/TPOT at multiple concurrencies.
10. Design a continuous-batching scheduler simulation.

Challenge:
- benchmark use_cache=True vs False on a tiny Transformers model
- build a discrete-event continuous-batching simulator
