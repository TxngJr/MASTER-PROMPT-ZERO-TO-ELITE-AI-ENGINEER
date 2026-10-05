# Chapter 47 Exercises — AI Agents and Tool Calling

Complete all 20. Keep code, configs, seeds, traces, metrics, and short conclusions.

## Level 1 — Recall
1. Define **tool schemas and bounded execution**.
2. Define **state/planning/memory** and its role.
3. Define **MCP concepts and tool interoperability** and one interoperability/architecture property.
4. Define **failure recovery, permissions and prompt-injection defense** and one failure mode.

## Level 2 — Understanding
5. Trace raw input/data through the system to final output and annotate state/shapes/artifacts.
6. State assumptions behind state/planning/memory and construct one violating example.
7. Derive/formalize the chapter's central probability/similarity/tokenization/objective/contract and verify a toy case.
8. Compare MCP concepts and tool interoperability and failure recovery, permissions and prompt-injection defense in quality, state, security, compute, and failure behavior.

## Level 3 — Coding
9. Implement one central operation/protocol from low-level primitives without hiding the mechanism in a high-level framework.
10. Add validation for schemas/IDs/shapes/ranges/masks/state transitions and untrusted external outputs.
11. Build a deterministic tiny end-to-end smoke test with known expected behavior.
12. Add invariant tests including one round-trip/reference/authorization/target-alignment check as appropriate.

## Level 4 — Debugging
13. Create a state/mask/tokenizer/ranking/schema bug that still runs; diagnose and regression-test the fix.
14. Create leakage, stale-index/state, or unsafe-permission behavior and demonstrate why the result is misleading/dangerous.
15. Trigger a context/tool-loop/numerical/resource failure and add a principled bound, timeout, validation, or stable formulation.
16. Profile preprocessing/retrieval/tool/model components separately and optimize the measured bottleneck.

## Level 5 — Challenge
17. Compare baseline and chapter method over at least three seeds/resamples or matched request sets with equal budgets.
18. Ablate one tokenizer/retrieval/state/architecture/training component and explain the mechanism.
19. Define a production security/reliability contract: versions, permissions, limits, timeouts, monitoring, fallback and rollback.
20. Write a research-style report with claim, baseline, protocol, resource budget, results, failure/security analysis, limitation, and next experiment.
