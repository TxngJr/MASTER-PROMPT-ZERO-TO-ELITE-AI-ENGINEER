# Batch 23 Integration Project — Reliable, Guarded & Interpretable AI Lab

Batch 23 connects distributed reliability, deterministic application safety controls and explanation validation.

## Part A — Distributed Reliability

~~~text
incoming requests
↓
bounded admission
↓
rendezvous placement
↓
workers
↓
retry/backoff
↓
circuit breaker
~~~

The lab reports:
- queue utilization
- accepted/rejected work
- deterministic replica placements
- retry schedule
- circuit state

## Part B — Guardrails

A mock student-facing system applies:

~~~text
authenticated group
↓
permission-aware retrieval
↓
tool allow-list
↓
write approval gate
↓
hard request budget
~~~

No real privileged or destructive tools are used.

The lab also computes safety block recall / false-positive rate from a held-out policy-decision example set.

## Part C — XAI

For a known linear function:

~~~text
input
↓
Integrated Gradients
↓
completeness check

same known additive contributions
↓
exact Shapley values
↓
attribution similarity
~~~

A second synthetic dataset demonstrates global permutation importance.

## Part D — Full System Gate

The full run passes only if:

~~~text
distributed capacity/reliability checks
AND
deterministic safety boundaries
AND
explanation completeness/consistency
~~~

This is deliberately stronger than checking model accuracy alone.

## Run

~~~bash
python integration-project-batch23/src/reliable_guarded_xai_lab.py --mode distributed
python integration-project-batch23/src/reliable_guarded_xai_lab.py --mode guardrails
python integration-project-batch23/src/reliable_guarded_xai_lab.py --mode xai
python integration-project-batch23/src/reliable_guarded_xai_lab.py --mode full
~~~

## Optional Ecosystem Labs

Ray:

~~~bash
python -m pip install -r requirements-batch23-ray.txt
~~~

Captum / SHAP:

~~~bash
# install PyTorch first with the official selector for your Fedora/NVIDIA setup
python -m pip install -r requirements-batch23-xai.txt
~~~

## Required Extensions

1. distributed worker failure injection
2. idempotent committed-result store
3. retry jitter and request deadlines
4. per-tenant bulkhead pools
5. guardrail regression dataset
6. permission-aware retrieval integration
7. safety classifier threshold sweep
8. Captum Integrated Gradients comparison
9. SHAP comparison
10. attribution stability/randomization sanity checks
