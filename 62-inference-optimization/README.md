# Chapter 62 — LLM Inference Optimization

## 1. Why Inference Is Different From Training

Autoregressive serving has two distinct phases:

~~~text
prompt
↓
PREFILL
many prompt tokens in parallel
↓
first generated token
↓
DECODE
one/few new tokens at a time
↓
response
~~~

These phases have different compute, memory and scheduling bottlenecks.

## 2. Learning Objectives

- distinguish prefill and decode
- derive KV-cache memory
- explain MHA/GQA/MQA cache savings
- explain dynamic/static/offloaded/quantized caches
- distinguish latency and throughput
- understand static/dynamic/continuous batching
- explain paged KV memory and fragmentation
- explain prefix caching and chunked prefill
- understand FlashAttention
- understand speculative decoding
- reason about distributed inference
- measure TTFT, TPOT, queue time and throughput

## 3. Prefill

During prefill, the prompt is processed and K/V states are produced for all prompt positions.

Important metric:

~~~text
TTFT = time to first token
~~~

Long prompts increase prefill work significantly.

## 4. Decode

After prefill, each new token produces a query that attends to cached keys/values.

Decode repeatedly reads model weights and KV cache and can become memory-bandwidth bound.

Important metric:

~~~text
TPOT = time per output token
~~~

## 5. KV Cache

Approximate dense cache storage:

~~~text
bytes =
2 × layers × batch × sequence_length
× kv_heads × head_dim × bytes_per_element
~~~

The factor 2 represents key and value.

## 6. MHA / GQA / MQA

~~~text
MHA: kv_heads = query_heads
GQA: kv_heads < query_heads
MQA: kv_heads = 1
~~~

Fewer KV heads reduce cache memory and decode bandwidth.

## 7. Long Context

Ordinary dense KV-cache memory grows approximately linearly with cached sequence length. Long context can reduce the number of concurrent requests even when model weights already fit.

## 8. Cache Strategies

Modern Transformers provides multiple strategies:

- Dynamic cache — grows with sequence
- Static cache — preallocates capacity and can help fixed-shape compilation
- Offloaded cache — trades GPU memory for CPU transfer overhead
- Quantized cache — lowers cache memory with extra quantization costs

Static cache can waste capacity/work when real sequence lengths are far below the configured maximum. citeturn379622search0turn379622search2

## 9. Latency vs Throughput

~~~text
latency = time per request
throughput = tokens/sec or requests/sec
~~~

Optimizing one can hurt the other.

## 10. Padding Waste

For sequence lengths L_i padded to max L:

~~~text
efficiency = sum(L_i) / (batch_size × max(L_i))
~~~

Length-aware batching can improve useful work.

## 11. Dynamic Batching

Collect requests inside a short scheduling window and form a batch. Larger windows may improve batch size but add queueing delay.

## 12. Continuous Batching

Instead of waiting for every sequence in a batch to finish, finished slots are replaced by new work during serving.

This improves utilization for variable output lengths. vLLM currently lists continuous batching as a core feature. citeturn912558search4

## 13. Paged KV Cache

Paged cache systems map logical request blocks to physical KV-memory blocks rather than requiring one large contiguous reservation.

~~~text
logical sequence blocks
↓ block table
physical KV blocks
~~~

This reduces allocator fragmentation and supports flexible sequence growth.

vLLM's current cache remains block-oriented, though its historical PagedAttention design document explicitly warns that it is not a literal description of all current kernel code. citeturn379622search3

## 14. Block Usage

For sequence length T and block size B:

~~~text
blocks = ceil(T/B)
allocated_slots = blocks × B
tail_waste = allocated_slots - T
~~~

Smaller blocks reduce tail waste but increase metadata and management overhead.

## 15. Prefix Caching

If requests share identical token prefixes, previously computed prefix KV blocks can be reused.

~~~text
common system/document prefix
↓ cached KV
new request suffix
↓ compute only missing suffix
~~~

Current vLLM uses hash-based automatic prefix caching. citeturn912558search5

## 16. Chunked Prefill

Very long prefills can monopolize compute and delay decode traffic. Chunked prefill splits prompt work so schedulers can interleave it with other requests.

## 17. FlashAttention

Standard dense attention conceptually forms an N×N score matrix. FlashAttention reorganizes exact attention into tiles to reduce expensive memory IO and avoid materializing the entire score matrix in high-bandwidth device memory.

Important:

> FlashAttention improves IO/memory behavior; standard dense attention still has quadratic token-pair arithmetic.

## 18. PyTorch SDPA

PyTorch scaled_dot_product_attention can dispatch to optimized backends depending on device, dtype and shape. Current torch.nn.attention exposes backend controls and FlashAttention implementations. citeturn379622search5

## 19. Speculative Decoding

Use a cheap draft method to propose multiple tokens, then let the target model verify them in parallel.

~~~text
draft proposes k tokens
↓
target verifies
↓
accept valid prefix
↓
advance multiple tokens when possible
~~~

Low acceptance can erase the speed benefit.

## 20. Speculation Correctness

Correct speculative algorithms preserve the target distribution under their sampling assumptions. Draft tokens must not simply bypass target verification.

## 21. Draft Methods

Possible proposal sources include:

- smaller draft model
- n-gram matches
- suffix/prefix heuristics
- learned multi-token heads

Current vLLM documents several speculative-decoding approaches. citeturn912558search4

## 22. Tensor Parallel Inference

Split large matrices across GPUs. This can fit larger models and combine bandwidth/compute, but introduces layer-level collectives.

## 23. Pipeline Parallel Inference

Split layer ranges across devices. Pipeline balance and scheduling influence end-to-end latency.

## 24. Data Parallel Serving

Replicate full model servers and route requests across replicas. This is often the simplest scaling path when one model copy fits per worker.

## 25. Prefix-Aware Routing

If cache reuse matters, routing requests with shared prefixes to the same replica can raise cache hit rate, creating a load-balance vs locality trade-off.

## 26. Queueing

End-to-end latency includes:

~~~text
queue time
+ prefill
+ decode
+ serialization/network
~~~

Fast kernels do not guarantee low request latency.

## 27. Metrics

- TTFT — time to first token
- TPOT / ITL — output-token latency
- output tokens/sec
- requests/sec
- queue time
- KV occupancy
- prefix-cache hit rate
- speculative acceptance rate

## 28. From Scratch

src/inference_math.py implements:

- kv_cache_bytes
- cache_reduction_ratio
- padded_batch_efficiency
- paged_block_usage
- prefix_cache_saved_tokens
- speculative_acceptance_rate
- average_tokens_per_target_verification
- bandwidth_lower_bound_seconds

## 29. Common Mistakes

1. prefill/decode mixed into one timing number
2. model weights measured while KV memory is ignored
3. throughput compared at different concurrency
4. static batching wastes padding
5. cache block confused with CUDA thread block
6. prefix-cache reuse based only on raw string equality
7. FlashAttention claimed to make dense attention linear
8. speculative tokens accepted without target verification
9. GPU kernel time confused with end-to-end latency
10. p50 optimized while p99 SLA is poor

## 30. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 31. Checklist

- [ ] prefill / decode
- [ ] KV cache
- [ ] GQA / MQA memory
- [ ] cache strategies
- [ ] continuous batching
- [ ] paged cache
- [ ] prefix caching
- [ ] FlashAttention
- [ ] speculative decoding
- [ ] distributed inference
- [ ] TTFT / TPOT / throughput

## 32. What's Next

Chapter 63 turns these inference primitives into production model-serving systems.