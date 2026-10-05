# Chapter 64 Exercises — Deploying AI

Complete all 20 with benchmark configs, code, plots and concise operational conclusions.

## Level 1 — Recall
1. Define **API/container packaging**.
2. Define **configuration/secrets** and its resource effect.
3. Define **Kubernetes deployment concepts** and one operational constraint.
4. Define **health checks, rollout and rollback** and one failure mode.

## Level 2 — Understanding
5. Trace an artifact/request from load through response/telemetry and identify resource ownership.
6. State assumptions behind configuration/secrets and construct one violation.
7. Derive a central byte/latency/throughput/quantization/SLO calculation and verify units.
8. Compare Kubernetes deployment concepts and health checks, rollout and rollback in latency, throughput, memory, complexity and resilience.

## Level 3 — Coding
9. Implement/simulate one core quantization/cache/scheduler/API/monitoring calculation with explicit state.
10. Add request/config/artifact validation plus strict limits and clear errors.
11. Build a deterministic single-request correctness baseline before optimization.
12. Add invariant tests for artifact identity, bounded inputs, quantization/cache state, health or metric calculations.

## Level 4 — Debugging
13. Introduce a cache/scale/batching/version/config bug that still serves responses; diagnose and regression-test.
14. Create timeout/cancellation/overload/rollout failure and verify cleanup or rollback.
15. Stress memory, queue, context/output length or concurrency until SLO failure; add a bounded mitigation.
16. Benchmark with warmup and repeated measurements; profile components and optimize the actual bottleneck.

## Level 5 — Challenge
17. Compare baseline vs optimized configuration with identical workload and quality target; report tail latency, throughput, memory and quality.
18. Ablate one cache/batching/quantization/deployment/monitoring component and explain the mechanism.
19. Write a production runbook covering deploy, health, alerts, incident triage, canary, rollback and artifact/config recovery.
20. Write an engineering report with claim, matched benchmark, workload, hardware, results, failure test, limitation and next experiment.
