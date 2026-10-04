# Chapter 63 — LLM Serving: Transformers, vLLM & llama.cpp

## 1. From Model to Service

A useful model becomes a production system only after request handling, scheduling, observability and failure management are added.

~~~text
client
↓
gateway / auth / rate limits
↓
model server
↓
scheduler
↓
inference engine
↓
model + KV cache
↓
streamed response
~~~

## 2. Learning Objectives

- distinguish local generation from online serving
- build a serving request/response contract
- understand streaming and cancellation
- explain readiness vs liveness
- measure TTFT / TPOT / throughput
- understand continuous batching and scheduling
- compare Transformers, vLLM and llama.cpp
- understand OpenAI-compatible APIs
- reason about GGUF serving
- expose metrics safely
- design rate limits/backpressure
- reason about autoscaling
- perform canary rollout
- identify serving security boundaries

## 3. Transformers Local Generation

Transformers is excellent for model loading, experimentation, custom generation loops and research applications.

~~~python
model.generate(...)
~~~

Current Transformers supports streaming through streamer objects so applications can surface generated text incrementally. citeturn131463search3turn131463search6

It is not automatically a high-throughput multi-tenant scheduler.

## 4. Serving Contract

A request should define:
- model
- prompt/messages
- max output tokens
- sampling settings
- stop conditions
- request ID
- optional structured-output/tool settings

Validate limits before scheduling expensive work.

## 5. OpenAI-Compatible APIs

Compatibility simplifies clients. Common routes include:

~~~text
/v1/models
/v1/completions
/v1/chat/completions
/v1/responses
/v1/embeddings
~~~

Exact support differs by runtime.

## 6. vLLM

vLLM currently targets high-throughput serving with continuous batching, chunked prefill, prefix caching, quantization, optimized attention, speculative decoding, distributed inference and streaming. It exposes OpenAI-compatible APIs including Completions, Chat Completions and Responses. citeturn912558search4turn131463search2

## 7. vLLM Launch Concept

A current-style command is:

~~~bash
vllm serve MODEL_ID
~~~

Pin runtime/model revisions in production rather than depending on moving defaults.

## 8. llama.cpp

llama.cpp focuses on efficient inference across CPU and accelerator backends and is especially important for GGUF and local deployment. Current project docs include an OpenAI-compatible server path. citeturn912558search2turn912558search6

## 9. llama.cpp Server

Current server docs include chat/completions, Responses-style APIs, embeddings, parallel decoding, continuous batching, monitoring and speculative decoding. Pin a release because exact commands/options can evolve. citeturn912558search3

## 10. GGUF Deployment

~~~text
model
↓ conversion / quantization
GGUF
↓
llama.cpp runtime
↓
CPU / GPU-offload inference
↓
HTTP/API client
~~~

Quantization changes memory, quality and speed trade-offs.

## 11. Streaming

Streaming returns incremental text/chunks as generation proceeds. It improves perceived latency but does not necessarily reduce total generation time.

Clients must handle partial text, tool-call deltas, finish reasons, disconnects and cancellation.

## 12. Cancellation

If a client disconnects or cancels:
- stop unnecessary decoding
- release KV blocks
- update metrics
- avoid zombie work

## 13. Request Limits

Validate:
- maximum prompt tokens
- maximum output tokens
- allowed models
- request size
- structured schema size

Reject impossible work before GPU allocation.

## 14. Timeouts

Different layers may enforce queue, TTFT, total-request and upstream gateway timeouts. Configure them coherently.

## 15. Liveness vs Readiness

Liveness asks whether restarting the process could help. Readiness asks whether the instance can currently accept production traffic.

A process can be live but not ready while a model is loading.

## 16. Health Checks

Avoid health checks that generate model text every few seconds. Use lightweight engine/readiness checks.

Current vLLM has a `/health` endpoint that returns success or engine-failure status. citeturn131463search11

## 17. Metrics

Track:
- requests / errors
- queue depth
- running requests
- prompt / generation tokens
- TTFT
- inter-token latency
- request latency
- KV-cache utilization
- prefix-cache hit rate
- tokens/sec

vLLM currently exposes Prometheus-compatible metrics at `/metrics`, including request, KV-cache and latency measurements. citeturn131463search0turn131463search8

## 18. Percentiles

Average latency hides tail failures. Track p50, p90, p95 and p99.

## 19. Capacity

Capacity depends on model, hardware, quantization, prompt/output distributions, concurrency, batching, KV memory and sampling. Benchmark realistic workloads.

## 20. Little's Law

For a stable system:

~~~text
average_concurrency ≈ arrival_rate × average_latency
~~~

If offered load exceeds sustainable throughput, queues grow.

## 21. Backpressure

When overloaded:
- reject early
- use bounded queues
- shed lower-priority work
- route to another replica

Unbounded queues transform overload into extreme latency or OOM.

## 22. Rate Limiting

Possible dimensions:
- requests/minute
- input tokens/minute
- output tokens/minute
- concurrent requests

Token-aware limits better represent LLM cost than request count alone.

## 23. Autoscaling

Signals can include queue depth, running requests, token throughput, KV utilization and latency SLOs. GPU utilization alone can be misleading.

## 24. Scale-Up Delay

Large models take time to load. Autoscaling may require warm pools, headroom and readiness gates.

## 25. Load Balancing

Possible strategies:
- round robin
- least requests
- least KV load
- prefix affinity
- model-aware routing

Prefix affinity may improve cache reuse but create hotspots.

## 26. Model Routing

A gateway can route by model ID, task, latency tier, context length or cost tier. Keep routing policy separate from engine internals.

## 27. Multi-Model / Adapter Serving

Challenges include GPU fragmentation, load/unload time, KV competition and adapter management. Adapters can be cheaper than a full model copy per task.

## 28. Canary Rollout

Send a small traffic percentage to a candidate version and compare errors, latency and quality. Expand or rollback based on explicit gates.

## 29. Shadow Traffic

Duplicate selected requests to a candidate without returning candidate outputs to users. Respect privacy/data policies.

## 30. Observability

Record request/trace ID, model/version, token counts, queue/prefill/decode timing, finish reason and error category.

Avoid logging sensitive prompt/output contents unless justified and protected.

## 31. API Security

Use network boundaries, authentication/authorization, reverse proxy/API gateway, TLS, rate limits, request validation and secret management.

Current vLLM explicitly warns that `--api-key` protects only certain path prefixes and should not be treated as the sole production security boundary. citeturn131463search1turn131463search4

## 32. Container / Orchestrator

A serving container should pin runtime/model revision and define startup, health/readiness, resources and graceful shutdown. Chapter 64 expands this into deployment.

## 33. Graceful Shutdown

1. stop accepting new work
2. drain/cancel active work
3. release resources
4. exit before hard kill

## 34. Benchmark Protocol

Record runtime version, model revision, quantization, hardware, prompt/output distributions, concurrency/request rate, warmup, duration and decoding settings.

## 35. From Scratch

src/serving_metrics.py implements:

- percentile
- request_latency
- throughput
- success_rate
- little_law_concurrency
- required_replicas
- token_rate_limit_cost
- canary_split

## 36. Runtime Selection

### Transformers
Best for research, custom pipelines and direct framework control.

### vLLM
Best for high-throughput GPU serving, scheduling and optimized inference.

### llama.cpp
Best for GGUF, CPU/local inference and hardware-flexible deployment.

These categories overlap.

## 37. Common Mistakes

1. raw inference server exposed directly to public internet
2. one server flag treated as the complete auth boundary
3. only one prompt length benchmarked
4. average latency reported without p99
5. unbounded queues
6. health check performs generation
7. cancellation does not release work
8. model revision not pinned
9. runtime/quantization changes during A/B test
10. prompt/output contents logged indiscriminately

## 38. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 39. Checklist

- [ ] Transformers generation
- [ ] vLLM
- [ ] llama.cpp / GGUF
- [ ] API contracts
- [ ] streaming / cancellation
- [ ] health / readiness
- [ ] metrics / percentiles
- [ ] backpressure / rate limits
- [ ] autoscaling
- [ ] canary rollout
- [ ] security boundary

## 40. What's Next

Batch 22 moves from serving engines into full deployment, MLOps and AI data engineering.