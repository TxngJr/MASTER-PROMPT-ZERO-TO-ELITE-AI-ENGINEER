# Chapter 66 Exercises — AI Data Engineering

Complete all 20 with code/configs, test fixtures, metrics and conclusions.

## Level 1 — Recall
1. Define **batch/stream pipelines**.
2. Define **data contracts and lineage** and its guarantee/goal.
3. Define **feature/training data quality** and one assumption.
4. Define **orchestration and backfills** and one limitation.

## Level 2 — Understanding
5. Trace one input/event/model/request through the full system and identify state/ownership/trust boundaries.
6. State assumptions behind data contracts and lineage and construct one violation.
7. Derive/formalize one key metric/invariant/objective and verify a toy calculation.
8. Compare feature/training data quality and orchestration and backfills in reliability, quality, cost and interpretability/safety.

## Level 3 — Coding
9. Implement/simulate one core data/system/safety/explanation/compression mechanism from explicit primitives.
10. Add schema/range/permission/state/version validation.
11. Build a deterministic fixture that demonstrates correct behavior and one expected failure.
12. Add invariant tests for idempotency/lineage/permissions/attribution/compression quality as appropriate.

## Level 4 — Debugging
13. Introduce a duplicate/retry/version/permission/attribution/compression bug that still looks plausible; diagnose and regression-test.
14. Create distribution shift, stale data/model state, or adversarial input and demonstrate why aggregate metrics hide it.
15. Trigger overload/resource/safety failure and add a bounded fallback or mitigation.
16. Profile cost by component and optimize the measured bottleneck without weakening correctness/safety.

## Level 5 — Challenge
17. Compare baseline and chapter method over multiple seeds/slices/failure fixtures with matched workload.
18. Ablate one reliability/safety/explainer/compression component and explain the mechanism.
19. Write a production incident/runbook contract with ownership, alerts, triage, rollback and evidence retention.
20. Write an engineering/research report with claim, baseline, protocol, stress/adversarial tests, resource/quality results, limitations and next experiment.
