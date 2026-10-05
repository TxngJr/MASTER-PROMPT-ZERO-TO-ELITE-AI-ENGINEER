# Chapter 64 Solutions — Deploying AI

Equivalent solutions are accepted when workload, artifact, state, resource and operational invariants remain correct.

## Level 1
### 1
Define **API/container packaging**, its state/artifact transformation, and the quality/resource trade-off.
### 2
Define **configuration/secrets** and state its effect on compute, memory or latency.
### 3
Explain **Kubernetes deployment concepts** and the limit/coordination needed to keep it safe.
### 4
Explain **health checks, rollout and rollback** and a concrete availability, quality, security or observability failure.

## Level 2
### 5
List artifact load, validation, queue/schedule, compute/cache, response and telemetry stages; mark CPU/GPU/shared state and cancellation/error transitions.
### 6
List assumptions and create a violating workload/config such as larger contexts, stale artifacts, unbounded concurrency or mismatched quantization axis.
### 7
Write the formula, substitute real numbers, preserve units, and compare with an independently computed result.
### 8
Compare with the same model/workload: p50/p95/p99, throughput, memory, implementation complexity, failure isolation and recovery behavior.

## Level 3
### 9
Expose core state explicitly and keep the simulation/reference small. For quantization include scale/zero-point math; for serving include queue/cache/request state.
### 10
Validate schema/ranges, artifact/version compatibility, max input/output, concurrency, timeouts and finite numeric values. Fail before expensive compute.
### 11
Use one fixed request/model/config and verify exact or tolerance-bounded output before adding batching/quantization/compilation/caching.
### 12
Test maximum/minimum bounds, artifact hash/config consistency, cache isolation, scale reconstruction, health/readiness transitions or metric math as applicable.

## Level 4
### 13
Use a deterministic fixture and state assertions to expose the bug. Repair the exact ownership/axis/version/config issue and add a regression test.
### 14
Force timeout/cancel/overload/bad rollout, verify resources are released and unhealthy versions stop receiving traffic, then prove rollback restores the baseline.
### 15
Increase one load dimension at a time, identify the limiting resource/SLO, apply a queue/concurrency/context/model/precision bound or autoscaling policy, and re-measure.
### 16
Warm up, synchronize accelerator timing when needed, collect multiple samples and percentiles, profile stages, optimize the hotspot and re-check output quality.

## Level 5
### 17
Freeze model, prompts, context/output lengths, hardware and quality threshold. Report all latency percentiles, throughput, memory and quality before/after.
### 18
Remove one optimization/operational component only and explain the measured effect using cache, queueing, precision, rollout or monitoring mechanics.
### 19
Include artifact/config IDs, deploy commands/process, preflight checks, dashboards/alerts, common failure symptoms, triage queries, canary gates, rollback and post-incident verification.
### 20
Make a falsifiable performance/reliability claim, use a matched workload, record hardware/runtime, show full quality/resource results and failure tests, state limitations and next measurement.
