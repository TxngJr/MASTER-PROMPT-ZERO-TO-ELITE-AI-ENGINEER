# Chapter 78 Exercises — Edge AI and TinyML

Complete all 20 with code, configs, seeds, plots and short conclusions.

## Level 1 — Recall
1. Define **resource-constrained inference**.
2. Define **quantization/deployment formats** and its role.
3. Define **sensor pipelines** and one uncertainty/resource/safety issue.
4. Define **latency/energy/memory budgeting** and one limitation.

## Level 2 — Understanding
5. Trace one end-to-end state/data/claim → model → output/action/evidence path.
6. State assumptions behind quantization/deployment formats and construct one violation.
7. Derive one central objective/dynamics/resource equation and verify a toy case.
8. Compare sensor pipelines and latency/energy/memory budgeting on quality, uncertainty, compute, safety and evidence.

## Level 3 — Coding
9. Implement/simulate one central operation from explicit low-level primitives.
10. Add validation for state/sensor/input/model/config/resource bounds.
11. Build a fixed-seed tiny task with a known expected trajectory/output/result.
12. Add invariant tests for state transitions, memory/resource math, split/tokenizer identity, or reproducibility metadata.

## Level 4 — Debugging
13. Create a model/state/sensor/data/config bug that still produces plausible output; diagnose and regression-test.
14. Create leakage, sim/real mismatch, unsafe action/resource assumption, or reproduction-protocol mismatch and fix it.
15. Stress horizon/input/context/latency/memory until failure and add a documented safety/resource bound.
16. Profile the complete pipeline and optimize only the measured bottleneck while preserving correctness/safety.

## Level 5 — Challenge
17. Compare baseline and chapter method over at least three seeds/scenarios with matched compute/data budgets.
18. Ablate one model/planning/sensor/reproduction/training component and explain the mechanism.
19. Write a deployment/publication gate with correctness, resource, safety, reproducibility, monitoring and rollback/limitations criteria.
20. Produce a final engineering/research report with claim, baseline, protocol, full results, failures, resource accounting, limitations and next experiment.
