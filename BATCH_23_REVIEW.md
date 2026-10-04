# Batch 23 Review — Chapters 67–69

## Chapters

- 67 — AI Distributed Systems
- 68 — AI Safety / Alignment / Guardrails
- 69 — Interpretability / XAI

## Distributed Systems Skills

- failure models
- stateless tasks vs stateful actors
- retries / retry budgets
- exponential backoff
- idempotency
- at-most-once / at-least-once / exactly-once boundaries
- bounded queues / backpressure
- circuit breakers / bulkheads
- sharding
- rendezvous hashing
- replication / quorum basics
- control plane vs data plane
- locality-aware scheduling
- stragglers
- heartbeats / leases
- distributed checkpoints
- distributed training/inference reliability

## Safety / Guardrails Skills

- alignment vs guardrails vs application security
- lifecycle risk management
- trust-boundary / threat modeling
- prompt-injection risk
- improper output handling
- least privilege
- excessive agency
- system-prompt limitations
- permission-aware RAG
- vector/embedding weaknesses
- tool-schema validation
- approval gates
- hard token/tool/step budgets
- supply-chain / poisoning concepts
- safety evaluation
- false-positive / false-negative trade-offs
- incident response

## Interpretability / XAI Skills

- local vs global
- intrinsic vs post-hoc
- coefficients / tree-importance caveats
- permutation importance
- finite-difference saliency
- Integrated Gradients
- baseline choice
- completeness
- Shapley values
- SHAP concepts
- occlusion
- counterfactual / PDP / ICE concepts
- stability / faithfulness / sanity checks
- attention limitations
- LLM token attribution
- probing
- activation patching
- ablation
- mechanistic-interpretability foundations
- explanation vs causal inference

## Implemented From Scratch

### Chapter 67
- exponential_backoff
- queue_utilization
- bounded_admission
- modulo_shard
- rendezvous_nodes
- quorum_overlap
- straggler_flags
- CircuitBreakerState
- circuit_breaker_record

### Chapter 68
- risk_priority
- threshold_decision
- budget_allows
- permission_decision
- retrieval_acl_filter
- validate_tool_arguments
- safety_confusion_counts
- safety_metrics

### Chapter 69
- finite_difference_gradient
- integrated_gradients
- completeness_gap
- occlusion_importance
- exact_shapley_values
- permutation_importance
- cosine_attribution_similarity

## Integration Project

The Batch 23 integration combines:

1. bounded distributed admission / rendezvous placement / retry / circuit breaking
2. permission-aware retrieval / tool approval / hard budgets
3. Integrated Gradients / exact Shapley / permutation explanations
4. one final system gate requiring reliability + safety boundary + explanation checks

## Current Ecosystem Audit

### Ray

Current Ray documentation describes:
- task / actor distributed primitives
- resource-aware scheduling
- data-locality preferences for tasks
- task/actor/object fault tolerance and retries
- Ray Data backpressure for controlling buffered work / object-store pressure

### OWASP GenAI Top 10:2025

Current categories include:
- Prompt Injection
- Sensitive Information Disclosure
- Supply Chain
- Data / Model Poisoning
- Improper Output Handling
- Excessive Agency
- System Prompt Leakage
- Vector / Embedding Weaknesses
- Misinformation
- Unbounded Consumption

OWASP explicitly treats system prompts as unsuitable places for secrets or as stand-alone security controls.

### NIST

NIST's AI Risk Management Framework resources include the Generative AI Profile for lifecycle risk management. The current NIST site notes that AI RMF 1.0 is under revision.

### Captum / SHAP

Current Captum provides generic PyTorch Integrated Gradients and convergence-delta support.

Current SHAP exposes a generic Explainer interface plus specialized explainers and represents explanations through Shapley-value-based methods.

Batch 23 includes a dedicated Captum smoke test that verifies Integrated Gradients on a known linear model.

## Methodology Audit

### Distributed Systems
- retry and idempotency are taught together
- overload is bounded rather than hidden behind unlimited queues
- exactly-once is scoped to an explicit system boundary
- placement is deterministic
- actor/state recovery requires explicit checkpoint semantics

### Safety
- model prompts never replace authorization
- sensitive retrieval is ACL-filtered before model use
- write actions require deterministic approval logic
- budgets are hard application controls
- safety metrics include benign cases and false positives
- examples use mock/non-destructive tool semantics

### Interpretability
- explanation methods are not presented as causal proof
- Integrated Gradients reports baseline/completeness
- exact Shapley implementation is capped to small feature counts due exponential cost
- permutation importance includes repeated shuffle variability
- attention/probe outputs are described as evidence requiring intervention, not causal conclusions

## Safety / Governance Audit

- no credentials are stored in prompts/examples
- tool authorization is deterministic
- write operations require approval in the integration lab
- retrieval authorization happens before content reaches the model
- untrusted model output is treated as data
- resource budgets limit unbounded consumption
- defensive red-team concepts are scoped to controlled regression testing

## Interpretation Audit

- high availability without idempotency can still duplicate side effects
- retries can amplify an outage
- a safe model cannot compensate for an overprivileged application
- blocking everything is not a valid safety system
- a visually plausible attribution can still be unfaithful
- decodable information in a probe does not prove causal use
- Shapley/IG values depend on formulation, baseline/background and output space

## Exit Gate

Before Chapter 70:

1. Batch 23 Core CI passes
2. Batch 23 Captum smoke passes
3. explain retries + idempotency together
4. demonstrate bounded overload behavior
5. route keys deterministically with rendezvous hashing
6. build a threat model with explicit trust boundaries
7. enforce least privilege and approval outside prompts
8. calculate safety recall and false-positive rate
9. implement Integrated Gradients and verify completeness
10. compute exact Shapley values on a small game
11. perform explanation stability/faithfulness checks
12. pass the full reliable + guarded + interpretable system gate
