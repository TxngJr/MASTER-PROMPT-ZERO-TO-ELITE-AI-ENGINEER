# Backpropagation — Exercises

## Level 1 — Recall
1. Define chain rule.
2. Define computational graph with a small example.
3. Explain reverse-mode autodiff.
4. Explain gradient accumulation.

## Level 2 — Understanding
5. Compare chain rule and computational graph.
6. Explain how gradient accumulation changes model behavior.
7. Explain a common limitation involving broadcasting gradients.
8. Explain the tradeoff introduced by gradient checking.

## Level 3 — Coding
9. Implement a small from-scratch example of chain rule.
10. Add deterministic tests for Exercise 9.
11. Implement the main computation behind reverse-mode autodiff.
12. Run a small seeded experiment using computational graph, gradient accumulation, and gradient checking.

## Level 4 — Analysis
13. Explain a common shape or data mismatch for this topic.
14. Explain one numerical-stability concern.
15. Explain the correct train, validation, and test workflow.
16. Measure runtime and memory of the main computation.

## Level 5 — Challenge
17. Derive the main input-to-output algorithm for Backpropagation.
18. Compare a scratch implementation with a trusted library on identical data.
19. Compare a baseline with a controlled change involving broadcasting gradients.
20. Summarize validation, reproducibility, monitoring, and one open research question.
