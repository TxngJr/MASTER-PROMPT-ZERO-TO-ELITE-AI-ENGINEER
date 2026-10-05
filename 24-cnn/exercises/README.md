# Chapter 24 Exercises — Convolutional Neural Networks

Complete all 20 exercises and retain code, calculations, plots, seeds, and concise conclusions.

## Level 1 — Recall
1. Define **convolution and receptive fields** and its role.
2. Define **padding/stride** and one numerical concern.
3. Define **pooling** and identify its state/parameters.
4. Define **CNN training and feature maps** and one limitation.

## Level 2 — Understanding
5. Trace the forward computation involving convolution and receptive fields and annotate tensor shapes.
6. State assumptions and failure modes for padding/stride.
7. Derive the central objective/gradient/update and verify a toy scalar/vector case by hand.
8. Compare pooling and CNN training and feature maps on abstraction, state, compute, memory, and debugging.

## Level 3 — Coding
9. Implement the chapter's central calculation using low-level tensor/NumPy operations rather than a high-level model wrapper.
10. Add shape/dtype/device/range validation and clear errors.
11. Build a fixed-seed tiny training or forward/backward smoke test with known expected behavior.
12. Add invariant-based tests, including one gradient/reference comparison where appropriate.

## Level 4 — Debugging
13. Create an axis/reduction/shape bug that runs but yields wrong results; diagnose with assertions.
14. Create a train/eval, gradient-state, checkpoint, or masking bug; fix it and add a regression test.
15. Trigger a numerical problem such as overflow, saturation, exploding gradients, or division by zero; implement a stable form or guard.
16. Profile one workload and optimize the actual bottleneck while proving numerical equivalence.

## Level 5 — Challenge
17. Compare two valid configurations across at least three seeds; report mean/variability and resource usage.
18. Run a one-factor ablation on a layer/objective/state choice and explain the mechanism.
19. Write a deployment contract covering accepted tensor shapes/dtypes, preprocessing, artifact version, device policy, limits, monitoring, and rollback.
20. Write a research note with claim, matched baseline, protocol, metrics, diagnostics, failure case, limitation, and next experiment.
