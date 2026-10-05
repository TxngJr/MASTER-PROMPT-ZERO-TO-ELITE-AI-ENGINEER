# Chapter 60 Solutions — LLM Evaluation

Equivalent solutions are accepted when the same masking, preference, parameter, evaluation, and reproducibility invariants hold.

## Level 1
### 1
Define **task/benchmark metrics** formally, identify trainable/frozen state or evaluated behavior, and explain its purpose.
### 2
Define **human and model-based evaluation**, state how examples/pairs/masks are constructed, and identify one assumption.
### 3
Explain **calibration/safety/error analysis** and a measurable optimization/evaluation risk.
### 4
Explain **contamination and statistical uncertainty** and a limitation involving data, reference/evaluator bias, optimization or generalization.

## Level 2
### 5
Write the full transformation from raw prompt/response/pair to tokens/masks/log-probabilities/scores and scalar objective/metric. Make reductions explicit.
### 6
List assumptions and create a violating case such as wrong template, prompt leakage, swapped pair or biased evaluator. Predict the effect.
### 7
Derive the objective/metric, define all symbols, calculate a tiny example, and state a range/sign/ordering invariant.
### 8
Compare with matched prompts and budget: supervision signal, reference requirements, memory/compute, stability, evaluator needs and failure modes.

## Level 3
### 9
Use low-level operations so low-rank math, masking, log-ratio or metric aggregation is visible. Match a hand-computed tiny case.
### 10
Assert valid token masks, chosen/rejected ordering, finite log-probabilities/scores, legal sequence lengths, and expected requires_grad/trainable parameter sets.
### 11
Fix seeds/fixtures and construct a tiny pair/task where one update should increase/decrease a known preference or loss. Verify the predicted direction.
### 12
Test prompt-token masking, frozen reference identity, adapter parameter count, preference-order symmetry/sign, or metric bounds as relevant.

## Level 4
### 13
Expose the bug using token-level loss masks, parameter-diff checks, or pair-swapping tests. Repair it and retain the regression test.
### 14
Use identical prompts, templates, decoding and evaluator settings before/after. Remove contaminated/tuned examples from test and report corrected results.
### 15
Inspect per-example loss/reward/log-ratio distributions and memory. Use stable log operations, clipping/regularization only when justified, smaller context/batch, or accumulation.
### 16
Measure tokenization, policy/reference forward, backward, and evaluation/judge separately. Optimize the actual bottleneck while preserving objective/metric tests.

## Level 5
### 17
Freeze prompt set, decoding, judge rubric and budget. Report all repeats/seeds plus confidence/variability and disagreement/failure slices.
### 18
Change exactly one data/objective/adapter/evaluator factor and explain the observed behavioral change mechanistically.
### 19
Gate releases on matched capability/safety/regression suites, resource limits and manual escalation criteria; version all artifacts and keep rollback-ready checkpoints.
### 20
Use a falsifiable claim, fair base model, frozen evaluator protocol, uncertainty, judge limitations, complete failures/negative results, and a next experiment targeting the main ambiguity.
