# Batch 24 Review — Chapters 70–72

## Chapters

- 70 — Model Compression
- 71 — Mixture of Experts
- 72 — Long Context / Memory

## Compression Skills

- unstructured / structured pruning
- global magnitude pruning
- sparsity / density
- masks vs real storage reduction
- iterative pruning
- knowledge distillation
- temperature-scaled soft targets
- KL distillation
- feature / relation distillation concepts
- pruning + quantization combinations
- end-to-end compression benchmarking

## MoE Skills

- router logits / softmax
- top-k routing
- selected-weight renormalization
- expert load
- capacity factor
- overflow/drop handling
- Switch-style load-balancing loss
- expert collapse
- total vs active parameters
- expert parallelism
- all-to-all communication
- serving memory/latency trade-offs

## Long Context / Memory Skills

- dense causal attention scaling
- sliding-window attention
- RoPE / scaling concepts
- positional interpolation/extrapolation
- KV-cache growth
- bounded cache for sliding/chunked layers
- chunking / overlap
- context packing
- context compression
- retrieval memory
- persistent memory policy
- Recall@K
- position-sensitive long-context evaluation

## Implemented From Scratch

### Chapter 70
- sparsity
- nonzero_count
- magnitude_prune
- structured_row_prune
- softmax_temperature
- kl_divergence
- distillation_loss
- compression_ratio

### Chapter 71
- stable_softmax
- top_k_router
- expert_load
- expert_capacity
- apply_capacity
- dropped_assignment_fraction
- switch_load_balance_loss
- active_parameter_fraction

### Chapter 72
- causal_dense_attention_pairs
- sliding_window_attention_pairs
- attention_pair_reduction
- kv_cache_bytes
- bounded_sliding_cache_tokens
- linear_rope_scaled_position
- chunk_ranges
- select_context_segments
- memory_score
- retrieval_recall_at_k

## Current Ecosystem Audit

### PyTorch

Current PyTorch documentation still includes torch.nn.utils.prune and an official pruning tutorial. Its knowledge-distillation tutorial demonstrates teacher soft targets mixed with ordinary supervised training.

### Transformers / RoPE

Current Transformers documentation exposes multiple RoPE variants, including default, linear, dynamic/NTK, YaRN, LongRoPE and Llama3-style scaling.

Current cache documentation also notes bounded growth for layers that use sliding-window or chunked attention after their configured window/chunk limit is reached.

### MoE

Current Mixtral documentation exposes router logits and auxiliary sparse-routing loss. Current DeepSpeed MoE documentation covers expert parallelism and communication-aware MoE training/inference.

## Methodology Audit

### Compression
- sparse masks are not claimed to guarantee smaller dense storage
- sparse weights are not claimed to guarantee latency speedup
- distillation temperature convention is explicit
- final compressed artifact must be reevaluated

### MoE
- total and active parameter counts are separate
- selected top-k gate weights are normalized
- expert capacity and overflow are explicit
- communication is included in scaling analysis

### Long Context
- RoPE extension is not treated as guaranteed long-context quality
- sliding-window attention is not mislabeled as global attention
- KV-cache memory is measured separately from attention pair count
- active context and persistent external memory are separate layers

## Exit Gate

Before Chapter 73:

1. Batch 24 Core CI passes
2. Batch 24 PyTorch smoke passes
3. explain unstructured vs structured pruning
4. reproduce temperature-scaled KD
5. show sparsity does not imply speedup
6. route top-k experts and calculate capacity
7. audit expert load and overflow
8. distinguish total vs active MoE parameters
9. derive dense vs sliding attention-pair growth
10. calculate long-context KV-cache cost
11. pack context under a token budget
12. evaluate external-memory Recall@K
13. pass the full compression + MoE + memory integration gate