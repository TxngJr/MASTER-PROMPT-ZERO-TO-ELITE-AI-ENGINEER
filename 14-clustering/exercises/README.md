# Chapter 14 Exercises — Clustering

Complete all 20. Keep code, calculations, plots, seeds, and short written conclusions.

## Level 1 — Recall
1. Define **K-Means** and state its purpose.
2. Define **DBSCAN** and give one concrete example.
3. Define **hierarchical clustering** and name one important hyperparameter or control.
4. Define **cluster validation** and state one limitation.

## Level 2 — Understanding
5. Explain how K-Means changes model behavior compared with a simpler baseline.
6. State the main assumptions behind DBSCAN and describe one violation.
7. Derive or justify the central objective/update/criterion used by this chapter; define every symbol.
8. Compare hierarchical clustering and cluster validation on bias, variance, compute, and interpretability.

## Level 3 — Coding
9. Implement a tiny independent reference for the chapter's central calculation without calling a high-level estimator.
10. Add input validation for shapes, dtypes, invalid hyperparameters, and empty inputs.
11. Add a deterministic experiment with a fixed seed and one synthetic dataset where the expected behavior is obvious.
12. Add at least two unit tests: one normal case and one edge case derived from a mathematical invariant.

## Level 4 — Debugging
13. Create a deliberately wrong implementation that produces plausible output; identify the bug using an invariant rather than visual inspection alone.
14. Construct a leakage or split bug, measure the inflated metric, then fix it and explain the difference.
15. Create a numerical/scale/pathological-input failure and add a guard or diagnostic.
16. Profile the implementation, find the dominant cost, and propose one optimization that preserves results.

## Level 5 — Challenge
17. Run a matched baseline vs chapter method across at least three seeds or data resamples; report mean and variability.
18. Perform a one-factor ablation on a central hyperparameter/component and explain the mechanism behind the observed change.
19. Design a production contract: accepted inputs, preprocessing, versioning, latency/memory budget, monitoring metric, and rollback condition.
20. Write a mini research note containing claim, method, baseline, controlled protocol, result table, failure case, limitation, and next experiment.
