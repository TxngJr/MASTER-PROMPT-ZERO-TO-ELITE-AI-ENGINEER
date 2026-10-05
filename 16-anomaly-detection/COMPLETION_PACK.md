# Completion Pack — Chapter 16: Anomaly Detection

Read after the main README. It standardizes the full teaching contract: prerequisites, mental model, derivation, code trace, experiments, debugging, performance, hardware, production, research, interview, and summary.

## Prerequisites
Complete the earlier dependency chain, run all chapter tests, and be able to inspect `src/` with a fixed seed and recorded environment.

## Mental Model
~~~text
data/state → representation → objective/dynamics → update rule → diagnostics → evaluation → decision
~~~
Core concepts: **statistical outliers, Isolation Forest, one-class methods, threshold calibration**.

## Core Theory Map
1. **statistical outliers** — explain the intuition, formal role, implementation, control knobs, and a failure mode.
2. **Isolation Forest** — explain the intuition, formal role, implementation, control knobs, and a failure mode.
3. **one-class methods** — explain the intuition, formal role, implementation, control knobs, and a failure mode.
4. **threshold calibration** — explain the intuition, formal role, implementation, control knobs, and a failure mode.

## Mathematics and Invariants
Derive the chapter's central equations, define every symbol and tensor shape, work a toy example, and verify with numerical checks. Use finite-value, shape, monotonicity, conservation/normalization, deterministic-seed, and reference-agreement invariants where applicable.

## Code Walkthrough
Trace inputs → preprocessing/state → forward/core calculation → update/backward → output/metric. Annotate shapes, dtypes, saved state, numerical guards, and complexity. Explain why each test exists.

## Visualization Lab
Create a data/state plot, a training/algorithm diagnostic, and a failure-case plot. Include labels, units, split, seed, and configuration.

## Experiment Design
Run a simple baseline, one-factor ablation, and stress test. For stochastic methods, use multiple seeds and report variability.

## Failure Cases and Debugging
Debug data/split first, then shapes/dtypes, objective/update equations, numerical stability, gradient correctness when relevant, evaluation mode, and finally hyperparameters.

## Performance Perspective
Profile the exact bottleneck and distinguish compute, memory, Python overhead, and data movement. Verify optimized code against the same invariant tests.

## Hardware-Aware Guidance
~~~yaml
Expected hardware:
  CPU: Ryzen 7 5825U-class or similar
  RAM: 16 GB recommended
  GPU: optional for tiny labs; useful from neural-network chapters onward
  VRAM: check locally; never assume an exact laptop-GPU capacity
  Dataset size: tiny/small synthetic or public datasets first
  Batch size: start 8–32 for neural-network smoke tests; reduce until memory is stable
~~~

## Production Perspective
Version data/preprocessing/model/config together, define input limits, monitor quality and resource metrics, persist reproducibility metadata, and design rollback behavior.

## Research Perspective
Use fair baselines, multiple seeds, uncertainty, ablations, matched budgets, and explicit limitations. A training curve alone is not evidence of a general claim.

## Common Mistakes
- mixing train/eval state;
- ignoring numerical stability;
- assuming a low training loss means correctness;
- failing to test gradients or update rules;
- comparing runs with unmatched seeds/budgets;
- hiding failed runs.

## Interview Questions
1. Explain statistical outliers from first principles.
2. Derive the key equation/update.
3. What assumption controls Isolation Forest?
4. Compare one-class methods with threshold calibration.
5. Which invariant finds the most dangerous bug?
6. How would you implement a tiny independent reference?
7. What is the dominant compute/memory cost?
8. Give a numerical-stability failure.
9. Design one ablation.
10. Give a production monitoring signal.

## Summary
Completion requires explanation, derivation, implementation, testing, debugging, profiling, controlled comparison, and application.
