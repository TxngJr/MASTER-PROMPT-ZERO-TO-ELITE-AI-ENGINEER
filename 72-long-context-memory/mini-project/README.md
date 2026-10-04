# Mini Project — Long-Context Memory Manager

Build a memory manager with:
- recent-turn buffer
- chunked document memory
- similarity/recency/importance scoring
- fixed active-context token budget
- retrieval provenance
- rolling summary slot

Evaluate:
- Recall@K
- context tokens used
- position sensitivity
- stale-memory errors
- conflicting-memory handling
- dense vs sliding-window attention/cache estimates
