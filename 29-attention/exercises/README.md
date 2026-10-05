# Chapter 29 Exercises — Attention

Complete all 20 exercises with code, calculations, plots, seeds, and short conclusions.

## Level 1 — Recall
1. Define **query/key/value projections** and its purpose.
2. Define **scaled dot-product attention** and identify its tensors/state.
3. Define **masking** and one failure symptom.
4. Define **multi-head attention** and one important implementation constraint.

## Level 2 — Understanding
5. Trace the full forward path and annotate tensor shapes.
6. Explain how scaled dot-product attention changes information or gradient flow.
7. Derive the central recurrence/objective/attention formula and verify a hand-computable case.
8. Compare masking and multi-head attention on optimization, expressivity, compute, and memory.

## Level 3 — Coding
9. Implement the chapter's central operation from low-level NumPy/tensor primitives.
10. Add mask/state/shape/dtype validation and fail-fast errors.
11. Create a fixed-seed tiny task that can be deliberately overfit or solved as a correctness smoke test.
12. Add tests for output shape, finite values, masking/state behavior, and one numerical/reference invariant.

## Level 4 — Debugging
13. Introduce a mask/axis/target-shift/state bug that still runs; diagnose and regression-test the fix.
14. Create an optimization/state bug such as stale state, incorrect detach, wrong train/eval mode, or objective sign; fix it.
15. Trigger exploding/vanishing/NaN/degenerate behavior and add a principled diagnostic or stability fix.
16. Profile sequence/input scaling, identify the true bottleneck, and optimize without changing semantics.

## Level 5 — Challenge
17. Run at least three seeds on a matched baseline and chapter model; report mean, variability, and resource usage.
18. Ablate one component such as a gate, head, residual, normalization, latent term, or stabilization trick.
19. Design a safe serving/training contract with maximum input length/size, batch limit, dtype/device policy, timeout/OOM behavior, and monitoring.
20. Write a research-style report with claim, matched baseline, protocol, results, diagnostics, failure cases, limitations, and next experiment.
