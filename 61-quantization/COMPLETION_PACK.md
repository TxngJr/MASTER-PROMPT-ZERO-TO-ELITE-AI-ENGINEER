# Completion Pack — Chapter 61: Quantization

Use after the main README to close the full inference/production teaching contract.

## Prerequisites
Complete model/evaluation/systems prerequisites. Record model/tokenizer/config, runtime/framework versions, hardware, traffic/workload shape, and benchmark method.

## Mental Model
~~~text
artifact → load/compile/index → request/batch → compute/memory path → response → telemetry → rollout/recovery
~~~
Core concepts: **PTQ/QAT concepts, symmetric/asymmetric scales, per-tensor/per-channel quantization, accuracy-memory-latency trade-offs**.

## Core Theory Map
1. **PTQ/QAT concepts** — explain the mechanism, resource trade-off, correctness/availability invariant, measurement method, and failure mode.
2. **symmetric/asymmetric scales** — explain the mechanism, resource trade-off, correctness/availability invariant, measurement method, and failure mode.
3. **per-tensor/per-channel quantization** — explain the mechanism, resource trade-off, correctness/availability invariant, measurement method, and failure mode.
4. **accuracy-memory-latency trade-offs** — explain the mechanism, resource trade-off, correctness/availability invariant, measurement method, and failure mode.

## Mathematics and Invariants
Derive byte/memory, quantization-error, throughput/latency, queueing or SLO calculations as appropriate. Track units explicitly. Verify bounded request sizes, deterministic artifact identity, quantization reconstruction tolerance, KV/cache shape, health-state transitions, and rollback safety.

## Code Walkthrough
Trace artifact loading → request validation → scheduling/batching → model compute → streaming/response → metrics/logs. Identify shared mutable state, cancellation/timeout paths, device memory ownership, and security boundaries.

## Visualization Lab
Plot latency percentiles vs load, throughput vs batch/concurrency, memory/KV usage, quantization error/quality, deployment health, drift or error-rate time series.

## Experiment Design
Run correctness baseline, one-factor optimization/rollout ablation, and overload/failure stress test. Keep model/input/output workload identical when comparing serving optimizations.

## Failure Cases and Debugging
Check artifact/version mismatch, unsafe loading, quantization scale/axis, cache/request ownership, unbounded queue/context/output, timeout/cancellation leaks, health checks, rollout config and metric attribution.

## Performance Perspective
Measure p50/p95/p99, throughput, time-to-first-token and inter-token latency where relevant. Warm up, separate prefill/decode, and profile CPU/GPU/network/data-store components.

## Hardware-Aware Guidance
~~~yaml
Expected hardware:
  CPU: enough for API/container/MLOps and quantization-analysis labs
  RAM: 16 GB recommended
  GPU: optional for tiny serving; useful for LLM inference benchmarks
  VRAM: inspect locally; model weights + KV cache + temporary buffers must fit
  Workload: synthetic bounded requests first, then representative traces
  Batch/concurrency: start 1 and increase while measuring latency, memory and quality
~~~

## Production Perspective
Use immutable artifact IDs, schema validation, least-privilege secrets, health/readiness checks, SLOs, dashboards/alerts, canary or staged rollout, and tested rollback.

## Research Perspective
When claiming an optimization, match model, quantization, prompts, context/output lengths, hardware and quality target. Report tail latency, memory and quality—not only throughput.

## Common Mistakes
- benchmarking different workloads;
- ignoring warmup/asynchrony;
- unbounded queues/context/output;
- loading arbitrary serialized objects;
- monitoring averages only;
- deploying without rollback;
- treating lower precision as free quality.

## Interview Questions
1. Explain PTQ/QAT concepts.
2. Derive a relevant memory/latency/quantization calculation.
3. What problem does symmetric/asymmetric scales solve?
4. Compare per-tensor/per-channel quantization and accuracy-memory-latency trade-offs.
5. Which serving/deployment invariant is critical?
6. How would you reproduce a benchmark fairly?
7. What dominates memory or tail latency?
8. Give an overload/rollout failure and mitigation.
9. Design an ablation.
10. What telemetry is mandatory before rollout?

## Summary
Production mastery means bounded behavior, measurable SLOs, reproducible artifacts, failure recovery, and quality-preserving optimization.
