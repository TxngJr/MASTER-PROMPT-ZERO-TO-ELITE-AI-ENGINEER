# Chapter 19 Exercises — Backpropagation

Complete all 20 exercises. Save code, calculations, plots, seeds, and concise conclusions.

## Level 1 — Recall
1. Define **computational graphs**.
2. Define **chain rule** and its main role.
3. Define **reverse-mode autodiff** and one important control/hyperparameter.
4. Define **gradient checking** and one limitation.

## Level 2 — Understanding
5. Explain the mechanism connecting computational graphs to the chapter's output.
6. State the assumptions behind chain rule and construct one violation.
7. Derive the central objective/update/recurrence; define every symbol and shape.
8. Compare reverse-mode autodiff and gradient checking in stability, compute, data needs, and failure modes.

## Level 3 — Coding
9. Implement a tiny independent version of the central calculation using Python/NumPy primitives.
10. Add validation for shape, dtype, range, empty input, and illegal hyperparameters.
11. Build a fixed-seed synthetic experiment where expected behavior is known before execution.
12. Add at least two invariant-based unit tests plus a comparison against a trusted tiny reference when feasible.

## Level 4 — Debugging
13. Introduce a wrong sign/axis/update/order bug that still runs; diagnose it using invariants.
14. Create a train/eval, leakage, or state-management bug; measure its effect and fix it.
15. Create a numerical extreme that causes overflow, underflow, instability, or degenerate behavior; implement a stable fix.
16. Profile one workload, identify the dominant operation or allocation, optimize it, and prove output equivalence.

## Level 5 — Challenge
17. Compare a baseline and chapter method over at least three seeds/resamples and report variability.
18. Run a one-factor ablation on a central component or hyperparameter and explain the mechanism.
19. Specify a production contract with inputs, artifact versions, resource limits, monitoring, and rollback.
20. Write a mini research report: claim, fair baseline, protocol, result table, diagnostics, limitation, and next experiment.
