# Chapter 73 Solutions — Reasoning Models

Equivalent answers are valid when compute budgets, state, tensor semantics and evaluation invariants are preserved.

## Level 1
### 1
Define **test-time compute**, its inputs/outputs/state, and why it is used.
### 2
Define **self-consistency/best-of-N** and explain what information/state it preserves or transforms.
### 3
Explain **verifiers and search** and its quality-vs-compute/token/sample trade-off.
### 4
Explain **process/outcome supervision and RLVR concepts** and a concrete routing/context/verifier/grounding/audio failure.

## Level 2
### 5
List every stage and annotate batch/sequence/expert/candidate/modality dimensions plus cache/router/verifier state.
### 6
List assumptions and create a violating example such as expert imbalance, longer-than-trained positions, correlated candidates, preprocessing mismatch or weak grounding.
### 7
Write the formula, define symbols/units, compute a toy case and state a normalization/range/resource invariant.
### 8
Compare under equal compute/token/data: representation, supervision, state, memory/latency, quality and failure modes.

## Level 3
### 9
Use explicit primitive operations. Verify against a hand-computed router/vote/cache/projection/feature example.
### 10
Validate expert/capacity indices, context/cache lengths, candidate counts, modality shapes/rates/ranges and finite values. Fail early.
### 11
Fix all randomness and use a tiny known fixture. Store configuration and assert the expected routing/vote/alignment/state result.
### 12
Test probability sums/load/capacity, byte/context calculations, pass@k/vote rules, projection shapes, or audio feature dimensions as relevant.

## Level 4
### 13
Use small deterministic fixtures and invariant assertions to isolate the bug. Repair the exact routing/state/shape rule and add a regression test.
### 14
Match candidate/token/context/model and preprocessing budgets exactly, then re-run. Explain why the original comparison confounded quality with extra compute or input differences.
### 15
Increase one resource dimension at a time, locate the limiting resource/quality point, impose a justified bound or approximation, and re-measure.
### 16
Time each subsystem and memory footprint separately. Optimize only the measured hotspot and re-run quality/invariant tests.

## Level 5
### 17
Freeze task/data, compute/token/candidate budget, model and evaluator. Report all seeds/samples, variability, quality and resource use.
### 18
Change one factor only and explain the effect through routing, context/state, verification or modality-representation mechanics.
### 19
Isolate per-request state/caches, version preprocessors/models/verifiers, cap context/candidates/modalities/output, monitor resource and quality/grounding/safety, and define fallback/rollback.
### 20
Use a falsifiable claim, fair baseline, explicit compute budget, complete results and failures, independent verification where possible, limitations and a targeted next experiment.
