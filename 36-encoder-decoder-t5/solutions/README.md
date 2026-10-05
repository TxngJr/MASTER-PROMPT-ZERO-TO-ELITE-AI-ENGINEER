# Chapter 36 Solutions — Encoder-Decoder and T5

Equivalent answers are accepted if they preserve the same formal, stochastic, and engineering invariants.

## Level 1
### 1
Define **encoder-decoder factorization** formally, explain its role, and identify the main state/input/output variables.
### 2
Define **cross-attention**, write its central rule or role in the system, and identify what information it uses.
### 3
Explain **teacher forcing and seq2seq loss** and name a measurable source of variance/instability plus a diagnostic.
### 4
Explain **text-to-text task formulation** and its quality/compute/sample-efficiency trade-off.

## Level 2
### 5
Write the full loop/path step by step. Include state/data creation, model/core operation, objective/value/score, update or iterative inference, and evaluation.
### 6
List assumptions, construct a violating example, and predict the failure before running. The final answer should name a test or plot that distinguishes the failure from random noise.
### 7
Derive the central equation, define symbols/shapes/probability terms, work a tiny numeric case, and state an invariant.
### 8
Compare under matched budgets: supervision signal, bias/variance, optimization stability, number of environment/sample/model steps, memory/latency, and evaluation difficulty.

## Level 3
### 9
Use primitive operations so the key update/operator remains visible. Match a hand calculation or trusted tiny reference.
### 10
Validate legal probability/range values, tensor/graph dimensions, terminal/mask semantics, schedule monotonicity or environment state as relevant. Fail early.
### 11
Fix seeds for environment/data/model/sampling, use a tiny known task, and verify expected behavior or learning direction. Store all config.
### 12
Test normal + edge behavior, including terminal/reset/mask/index/normalization invariants. Compare against a tiny trusted calculation where feasible.

## Level 4
### 13
Create one plausible bug, detect it with an invariant rather than score alone, repair the minimum cause, and add a regression test.
### 14
Show why one seed/sample/rollout or incorrect reset gives an unreliable comparison. Re-run with matched budgets and multiple seeds and report the change.
### 15
Inspect losses/rewards/gradient norms/scores/sample statistics. Apply stable math, clipping/normalization or step-size changes only when justified by the diagnosis.
### 16
Time each subsystem separately. Optimize the dominant environment loop, graph operator, model kernel, or iterative sampler and verify equivalent semantics/results within tolerance.

## Level 5
### 17
Use identical budgets/protocols and report every seed, mean, spread/interval, resource usage, and failures. Do not select the best trajectory/sample.
### 18
Change one factor only and connect its effect to the formal role of that component.
### 19
Bound inputs, iterations/rollout length/sampling steps, batch/concurrency and memory; version all artifacts/config; monitor quality/reward/drift/resource use; define timeout and rollback behavior.
### 20
Use a falsifiable claim, fair baseline, frozen protocol, full distributions/results, failure/negative evidence, limitations, and a next experiment targeted at the largest uncertainty.
