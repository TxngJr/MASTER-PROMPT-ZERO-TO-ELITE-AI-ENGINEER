# Chapter 35 Exercises — GPT and Decoder Models

Complete all 20 with code/calculations/plots/configuration records.

## Level 1 — Recall
1. Define **causal language modeling**.
2. Define **next-token loss** and its training/inference role.
3. Define **sampling/decoding** and one important assumption.
4. Define **KV-cache-aware generation concepts** and one cost/failure trade-off.

## Level 2 — Understanding
5. Trace raw input to final representation/output and annotate shapes.
6. Explain how next-token loss changes the learning signal or contextual information.
7. Derive the central objective/probability/similarity calculation and work a tiny example.
8. Compare sampling/decoding and KV-cache-aware generation concepts in representation, compute, data, and evaluation.

## Level 3 — Coding
9. Implement the chapter's core token/embedding/attention/objective/decoding calculation from low-level operations.
10. Add validation for IDs/shapes/masks/special tokens/ranges and finite values.
11. Build a fixed-seed tiny dataset/task with an obvious expected behavior.
12. Add tests for token/shape/mask/normalization/target-shift invariants.

## Level 4 — Debugging
13. Create a padding/causal-mask, special-token, or target-shift bug and diagnose it with a tiny sequence.
14. Create tokenizer/data-split or train/eval leakage and show the misleading metric before fixing it.
15. Trigger long-sequence/vocabulary/numerical/resource stress and implement a principled bound or mitigation.
16. Profile preprocessing and model separately; optimize the measured bottleneck and preserve output semantics.

## Level 5 — Challenge
17. Compare a baseline and chapter method across at least three seeds/resamples with matched token/data budgets.
18. Ablate one representation/model/decoding component and explain the mechanism.
19. Define a production contract for tokenizer/model versions, max input/output length, batch/concurrency, monitoring, and rollback.
20. Write a mini research report with claim, baseline, protocol, contamination controls, result table, error analysis, limitations, and next experiment.
