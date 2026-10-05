# Completion Pack — Chapter 03: Scientific Python

This pack closes the chapter-level teaching requirements that are easy to miss when theory, code, tests, and projects are split across files. Read it **after the main README and before the mini-project**.

## Prerequisites

- Complete the earlier chapters in the dependency chain.
- Be able to run the chapter tests and read the reference implementation in `src/`.
- Keep a reproducible environment: code commit, Python version, dependency versions, seed, dataset identity, and hardware notes.

## Mental Model

~~~text
problem
→ representation / assumptions
→ objective or rule
→ algorithm / implementation
→ diagnostics
→ evaluation
→ failure analysis
→ production or research decision
~~~

For this chapter, the core concepts are: **NumPy vectorization, Pandas dataframes, Matplotlib, numerical reproducibility**.

## Core Theory Map

1. **NumPy vectorization** — explain what it represents, which assumptions make it valid, what observable behavior it creates, and one case where it fails.
2. **Pandas dataframes** — explain what it represents, which assumptions make it valid, what observable behavior it creates, and one case where it fails.
3. **Matplotlib** — explain what it represents, which assumptions make it valid, what observable behavior it creates, and one case where it fails.
4. **numerical reproducibility** — explain what it represents, which assumptions make it valid, what observable behavior it creates, and one case where it fails.

A complete explanation must connect **intuition → formal definition → implementation → observable evidence**. Naming an API is not a substitute for explaining the mechanism.

## Mathematics and Invariants

For every important equation or rule in the chapter:

1. define every symbol and its domain;
2. annotate scalar/vector/matrix/tensor shapes;
3. derive the result from the previous line instead of memorizing it;
4. verify a tiny hand-computable example;
5. implement a numerical check;
6. state at least one invariant that tests correctness.

Useful generic invariants include finite outputs, shape preservation, probability normalization where applicable, monotonic loss behavior on a toy case, deterministic output under fixed seed, and train/test separation.

## Code Walkthrough

Use the reference implementation in `src/` as a trace exercise:

1. identify public inputs and outputs;
2. follow validation and preprocessing;
3. trace the central algorithm line by line;
4. write down every intermediate shape/state;
5. identify numerical-stability guards;
6. locate complexity bottlenecks;
7. map each test to the behavior it protects;
8. modify one small component and predict the effect before running it.

## Visualization Lab

Create at least three diagnostics appropriate to this chapter:

- **data/state view** — inspect inputs, distributions, shapes, or transitions;
- **learning/algorithm view** — plot objective, error, convergence, or intermediate quantities;
- **failure view** — visualize one edge case or violated assumption.

A figure is only useful when its axes, units, split, seed, and interpretation are recorded.

## Experiment Design

Run three controlled experiments:

1. **baseline** — simplest correct configuration;
2. **ablation** — change exactly one factor;
3. **stress test** — push an assumption or resource limit until behavior degrades.

Report configuration, seed, metric, uncertainty when relevant, hardware, and an explanation of *why* the result changed.

## Failure Cases and Debugging

Debug in this order:

1. data identity and split;
2. shapes / dtypes / units;
3. objective or algorithm definition;
4. numerical stability;
5. gradients or update rule when relevant;
6. evaluation mode and metric;
7. random seed / nondeterminism;
8. performance and memory.

Do not tune hyperparameters to hide a correctness bug.

## Performance Perspective

Measure before optimizing. Separate algorithmic complexity from implementation overhead. Record dataset/input size, wall-clock measurement method, memory pressure, and the exact configuration being compared.

## Hardware-Aware Guidance

~~~yaml
Expected hardware:
  CPU: modern x86-64 CPU; Ryzen 7 5825U-class is sufficient for chapter-scale labs
  RAM: 8 GB minimum; 16 GB recommended
  GPU: not required for this chapter
  VRAM: not required
  Dataset size: use tiny/small datasets first; scale only after correctness tests pass
  Batch size: use the smallest value that makes the example clear; batch size is not a correctness target
~~~

## Production Perspective

Before shipping a system based on this chapter, define input validation, versioned data/configuration, tests, observability, rollback behavior, and a failure policy. A notebook result is not a deployment contract.

## Research Perspective

A research-quality claim needs a baseline, controlled comparison, seeds/uncertainty when stochastic, an ablation or diagnostic, compute/data accounting, and limitations. Prefer falsifiable claims over narrative explanations.

## Common Mistakes

- memorizing terminology without being able to derive or implement it;
- reporting one metric without inspecting failure cases;
- using the test set during development;
- changing several variables in one experiment;
- treating library defaults as theory;
- omitting seed, versions, or data identity;
- optimizing speed before establishing correctness.

## Interview Questions

1. Explain NumPy vectorization from first principles.
2. What assumptions are required for Pandas dataframes?
3. Compare NumPy vectorization and Matplotlib in terms of use and failure modes.
4. Which invariant would you test first and why?
5. How would you detect data leakage or evaluation contamination?
6. Describe a minimal from-scratch implementation.
7. What is the dominant time or memory cost?
8. Give one production failure and a mitigation.
9. Design an ablation that tests a central claim.
10. What evidence would convince you that the implementation is correct?

## Summary

You are finished only when you can **explain, derive, implement, test, debug, measure, compare, and apply** the chapter concepts without relying on hidden library behavior.
