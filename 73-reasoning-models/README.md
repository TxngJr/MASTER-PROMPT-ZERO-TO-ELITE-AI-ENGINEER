# Chapter 73 — Reasoning Models

## 1. What Changes at Reasoning Time?

A reasoning-oriented system may spend additional inference compute before committing to one answer.

~~~text
problem
↓
one or many candidate solutions
↓
verification / voting / search
↓
final answer
~~~

The key engineering question is not only accuracy, but how much extra compute/tokens/latency were required.

## 2. Learning Objectives

- distinguish training-time and test-time compute
- understand chain-of-thought concepts without treating traces as ground-truth internal reasoning
- implement self-consistency voting
- implement best-of-N with a verifier
- calculate pass@k
- allocate token budgets across samples
- distinguish outcome and process supervision
- understand verifiers/reward models
- understand search over candidate reasoning states
- reason about GRPO/RLVR-style post-training concepts
- evaluate reasoning under accuracy/cost/latency constraints
- understand reasoning distillation

## 3. Chain-of-Thought as an Interface

Intermediate text can help decompose a task and provide training/evaluation signals, but a generated trace should not automatically be interpreted as a faithful transcript of hidden computation.

Evaluate the observable behavior:
- final correctness
- consistency
- verifier agreement
- interventions/search outcomes
- cost

## 4. Test-Time Compute

Instead of one sample:

~~~text
x -> y
~~~

generate multiple candidates:

~~~text
x -> y1
x -> y2
...
x -> yN
~~~

Then aggregate/select them.

## 5. Self-Consistency

Sample multiple candidate solutions and choose the most frequent final answer.

~~~text
confidence = majority_count / samples
~~~

This can improve robustness when independent samples make different reasoning errors.

## 6. Majority Vote Limitations

Majority can still be wrong when:
- samples are highly correlated
- model has a systematic misconception
- parsing final answers is unreliable
- temperature/sample settings are poor

More samples are not automatically better.

## 7. Best-of-N

Generate N candidates and score each with a verifier/reward model.

~~~text
candidate_i -> verifier score_i
select argmax score_i
~~~

Verifier quality becomes a new failure point.

## 8. Pass@K

If N samples include c correct samples, a common unbiased combinatorial estimator is:

~~~text
pass@k = 1 - C(N-c, k) / C(N, k)
~~~

When fewer than k incorrect samples exist, pass@k is 1.

## 9. Outcome Supervision

Supervise only whether the final answer/result is correct.

Pros:
- easier labels
- objective automatic checks possible

Cons:
- sparse credit assignment
- incorrect intermediate steps can accidentally reach a correct result

## 10. Process Supervision

Score/check intermediate steps.

Possible signals:
- step validity
- invariant preservation
- subgoal completion
- tool result verification

Process labels are more expensive and can themselves be noisy.

## 11. Verifiers

A verifier predicts whether a candidate/step satisfies a criterion.

Examples:
- exact symbolic check
- unit test
- constraint checker
- learned reward/verifier model

Prefer deterministic verification where the domain permits it.

## 12. Search

Reasoning can be framed as search:

~~~text
state
├── candidate step A
├── candidate step B
└── candidate step C
~~~

Search strategies include:
- beam search
- best-first
- tree search
- graph search
- MCTS-style methods

The state representation and verifier define what search can discover.

## 13. Compute Budget

One total token budget can be divided across:
- number of candidates
- tokens/candidate
- verification
- search depth

Spending all budget on one very long trace may be worse than several shorter independent attempts.

## 14. Search Efficiency

Track:

~~~text
solve rate
solved / 1k generated tokens
tokens / solved task
latency / solved task
~~~

Reasoning benchmark results without compute budgets are incomplete.

## 15. Reward Hacking / Verifier Exploitation

If selection optimizes a learned verifier, candidates may exploit verifier weaknesses rather than genuinely improve correctness.

Mitigations:
- hidden test checks
- diverse verifiers
- adversarial evaluation
- exact execution/checking where possible
- distribution-shift monitoring

## 16. RLVR / GRPO Concepts

Reinforcement learning with verifiable rewards uses rewards that can be checked automatically for suitable tasks.

Group-relative methods compare sampled outputs from the same prompt rather than requiring the exact same critic design as PPO.

Current Hugging Face TRL includes GRPOTrainer alongside SFT, DPO and reward-modeling trainers. citeturn537494search5

## 17. Reasoning Distillation

A capable teacher can produce candidate solutions/signals used to train a smaller student.

Possible targets:
- final answers
- selected traces
- verifier-filtered traces
- intermediate structured states

Distillation inherits any errors/biases in teacher traces and filters.

## 18. Reasoning Evaluation

Evaluate:
- final-answer accuracy
- pass@k
- self-consistency
- verifier precision/recall
- tokens generated
- latency
- cost
- robustness under reordered/perturbed prompts

## 19. From Scratch

`src/reasoning.py` implements:
- majority_vote
- self_consistency_confidence
- best_of_n
- pass_at_k
- allocate_sample_budget
- verifier_accuracy
- search_efficiency
- weighted_vote

## 20. Common Mistakes

1. chain-of-thought text treated as guaranteed faithful internal reasoning
2. reasoning models compared at different token budgets
3. best-of-N uses an unvalidated verifier
4. self-consistency samples are nearly identical
5. pass@k calculated as plain accuracy
6. verifier score mistaken for correctness
7. unlimited reasoning tokens
8. latency ignored
9. process labels assumed perfect
10. one benchmark treated as universal reasoning ability

## 21. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 22. Checklist

- [ ] test-time compute
- [ ] self-consistency
- [ ] best-of-N
- [ ] pass@k
- [ ] verifier
- [ ] outcome/process supervision
- [ ] search
- [ ] compute budget
- [ ] RLVR/GRPO concepts
- [ ] cost-aware evaluation

## 23. What's Next

Chapter 74 combines language reasoning with visual inputs and grounding.