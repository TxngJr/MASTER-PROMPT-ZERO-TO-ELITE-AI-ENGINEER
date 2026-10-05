# Chapter 39 Exercises — Generative AI Foundations

Complete all 20 exercises. Save code, calculations, plots, seeds, trajectories/samples, and concise conclusions.

## Level 1 — Recall
1. Define **likelihood-based models**.
2. Define **latent-variable models** and its update/role.
3. Define **adversarial generation** and one source of variance/instability.
4. Define **diffusion/score modeling overview** and one important trade-off.

## Level 2 — Understanding
5. Trace the algorithm from input/state through objective/update to output/action/sample.
6. State assumptions behind latent-variable models and construct one failure case.
7. Derive the central equation/objective/update and verify a tiny hand calculation.
8. Compare adversarial generation and diffusion/score modeling overview in supervision, stability, compute/sample cost, and evaluation.

## Level 3 — Coding
9. Implement a minimal independent version of the chapter's central operator/update with low-level primitives.
10. Add validation for shapes, probabilities/ranges, masks/terminal state, graph indices, or schedule parameters as appropriate.
11. Build a fixed-seed toy task whose expected behavior can be checked directly.
12. Add invariant tests plus one tiny reference comparison.

## Level 4 — Debugging
13. Introduce a terminal/mask/index/target/sign bug that still runs; diagnose and regression-test the fix.
14. Create a stochastic-evaluation bug such as one-seed reporting, bad reset, or mismatched sampling budget; fix the protocol.
15. Trigger numerical/optimization instability and add a principled diagnostic/mitigation.
16. Profile environment/data/model/sampling components separately and optimize the measured bottleneck without changing semantics.

## Level 5 — Challenge
17. Compare a baseline and chapter method over at least three seeds with matched data/rollout/sample budgets.
18. Ablate one central component and explain the mechanism behind the result.
19. Design a bounded production contract with input/environment limits, iterative-compute cap, artifact versions, monitoring, and rollback.
20. Write a research-style report with claim, baseline, protocol, full stochastic results, failure analysis, limitations, and next experiment.
