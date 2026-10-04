# Batch 21 Review — Chapters 61–63

## Chapters

- 61 — Quantization: INT8 / INT4 / GPTQ / AWQ / GGUF
- 62 — Inference Optimization
- 63 — LLM Serving

## Quantization Skills

- symmetric / asymmetric quantization
- scale / zero point
- per-tensor / per-channel / grouped quantization
- quantization error
- clipping / calibration
- weight-only vs activation quantization
- PTQ vs QAT
- GPTQ / AWQ concepts
- GGUF vs quantization-algorithm distinction
- raw storage / metadata accounting

## Inference Optimization Skills

- prefill vs decode
- TTFT / TPOT
- KV-cache memory
- GQA / MQA cache savings
- Dynamic / Static / Offloaded / Quantized caches
- padding efficiency
- dynamic / continuous batching
- paged KV memory
- prefix caching
- chunked prefill
- FlashAttention / SDPA
- speculative decoding
- tensor / pipeline / data-parallel inference

## Serving Skills

- local generation vs online serving
- Transformers / vLLM / llama.cpp roles
- OpenAI-compatible API concepts
- streaming / cancellation
- liveness / readiness
- metrics / percentiles
- Little's Law
- bounded queues / backpressure
- token-aware rate limits
- autoscaling
- model/prefix-aware routing
- canary / shadow rollouts
- security boundaries
- reproducible serving benchmarks

## Implemented From Scratch

### Chapter 61
- signed_integer_range
- symmetric_quantize
- symmetric_dequantize
- asymmetric_quantize
- asymmetric_dequantize
- per_channel_symmetric_quantize
- grouped_symmetric_quantize
- quantization_mse
- ideal_storage_bytes
- compression_ratio

### Chapter 62
- kv_cache_bytes
- cache_reduction_ratio
- padded_batch_efficiency
- paged_block_usage
- prefix_cache_saved_tokens
- speculative_acceptance_rate
- average_tokens_per_target_verification
- bandwidth_lower_bound_seconds

### Chapter 63
- percentile
- request_latency
- throughput
- success_rate
- little_law_concurrency
- required_replicas
- token_rate_limit_cost
- canary_split

## Integration Project

Four modes:

1. INT8/INT4/per-channel quantization quality
2. KV-cache / paging / prefix / speculation capacity planning
3. serving SLO / replica / canary planning
4. PyTorch weight-only reconstruction smoke

## Current Ecosystem Audit

### Transformers

Current Transformers documents:
- multiple quantization integrations including bitsandbytes, GPTQ and AWQ
- multiple KV-cache strategies including Dynamic, Static, Offloaded and Quantized caches
- generation streaming APIs

### vLLM

Current vLLM documents:
- continuous batching
- chunked prefill
- prefix caching
- quantization backends
- optimized attention kernels
- speculative decoding
- distributed inference
- OpenAI-compatible APIs
- Prometheus-compatible /metrics
- /health

### llama.cpp

Current llama.cpp documents:
- GGUF inference
- CPU / accelerator backends
- quantized models
- OpenAI-compatible server
- parallel decoding
- continuous batching
- monitoring
- speculative decoding

## Methodology Audit

### Quantization
- raw low-bit storage is separated from metadata/runtime buffers
- GGUF is not mislabeled as GPTQ/AWQ
- low-bit storage is not assumed to imply equal-bit arithmetic
- downstream output/task quality is required in addition to tensor MSE

### Inference
- prefill and decode are measured separately
- KV memory includes K and V explicitly
- GQA cache savings are derived from KV-head reduction
- paged block tail waste is measured
- FlashAttention is described as IO-aware exact attention, not linear dense attention
- speculative speedup is conditioned on acceptance/overhead

### Serving
- queue time is included in end-to-end latency
- p50 and tail percentiles are separate
- autoscaling uses headroom rather than peak theoretical capacity
- canary traffic is explicit
- serving authentication is treated as a layered boundary

## Security Audit

Current vLLM documentation warns that --api-key does not authenticate every route. Production designs should place inference behind a real reverse proxy/network/auth boundary rather than exposing a raw model server directly.

## Interpretation Audit

- INT4 does not guarantee 4x speedup vs FP16
- smaller model files do not guarantee lower end-to-end latency
- static cache may trade memory/work for compile-friendly shapes
- high throughput can coexist with poor p99 latency
- prefix caching depends on compatible token/config identity
- GPU utilization alone is not an autoscaling policy
- benchmark numbers without workload/runtime/model revisions are not comparable

## Exit Gate

Before Chapter 64:

1. Batch 21 Core CI passes
2. Batch 21 PyTorch smoke passes
3. derive symmetric/asymmetric quantization
4. compare per-tensor/channel/group error
5. estimate weight + KV memory independently
6. explain prefill vs decode
7. calculate paged-block waste
8. explain FlashAttention accurately
9. measure speculative acceptance
10. calculate p50/p99 and replica capacity
11. design bounded queue/backpressure
12. record serving benchmark manifest
