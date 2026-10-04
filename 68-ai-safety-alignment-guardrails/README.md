# Chapter 68 — AI Safety, Alignment & Guardrails

## 1. Safety Is Defense in Depth

No single prompt, classifier or model setting can make an AI application safe.

~~~text
user / external content
↓
input validation
↓
retrieval / context authorization
↓
model
↓
tool permission boundary
↓
output validation
↓
human approval where needed
↓
action / response
~~~

## 2. Learning Objectives

- distinguish model alignment from application security
- build an AI threat model
- understand prompt-injection risk
- understand improper output handling
- enforce least privilege for tools
- understand system-prompt leakage limitations
- secure RAG/vector access
- prevent unbounded resource consumption
- validate tool arguments and outputs
- design approval gates
- build risk registers
- design safety evals and red-team test cases
- reason about false positives/false negatives
- apply NIST-style lifecycle risk management concepts

## 3. Alignment vs Guardrails vs Security

### Alignment
Training/post-training so model behavior better matches desired objectives/preferences.

### Guardrails
Runtime policies, validators and decision layers around a model.

### Application Security
Authentication, authorization, isolation, secrets, network controls, secure software engineering.

These overlap but are not interchangeable.

## 4. Risk Management

NIST AI RMF frames AI risk management across the lifecycle rather than as a single pre-deployment check.

Current NIST materials include a Generative AI Profile and note that AI RMF 1.0 is under revision.

Use a risk process:
1. identify assets and users
2. identify hazards/threats
3. estimate likelihood/impact
4. choose mitigations
5. test controls
6. monitor residual risk

## 5. Threat Modeling

For each component identify:
- trusted inputs
- untrusted inputs
- sensitive assets
- privileges
- external side effects
- trust boundaries

Example components:
- user prompt
- uploaded document
- RAG store
- system instruction
- model
- tool/API
- database
- output renderer

## 6. OWASP GenAI Risks

OWASP's current 2025 LLM/GenAI Top 10 includes categories such as:
- prompt injection
- sensitive information disclosure
- supply-chain vulnerabilities
- data/model poisoning
- improper output handling
- excessive agency
- system prompt leakage
- vector/embedding weaknesses
- misinformation
- unbounded consumption

Use these as threat-model prompts, not as a substitute for system-specific analysis.

## 7. Prompt Injection

Untrusted text can contain instructions that conflict with application intent.

Sources include:
- direct user content
- retrieved documents
- webpages
- tool outputs
- messages from other agents

Key principle:

> Treat untrusted content as data, not as authorization.

## 8. System Prompt Is Not a Security Boundary

A system prompt can guide model behavior but should not contain secrets or be the only mechanism enforcing permissions.

Authorization must be enforced in deterministic application code.

## 9. Least Privilege

A model should have only the tools/permissions necessary for the current task.

Prefer:
- read-only tools by default
- narrow scopes
- per-user authorization
- short-lived credentials
- allow-listed operations

## 10. Excessive Agency

Giving a model unnecessary actions, permissions or autonomy increases blast radius when the model is wrong or manipulated.

Reduce:
- number of enabled tools
- write permissions
- autonomous step count
- accessible data

## 11. Tool Schema Validation

Tool calls should pass deterministic validation:
- tool exists
- caller authorized
- argument types valid
- values within allowed ranges
- request budget not exceeded

The model's confidence is not authorization.

## 12. Approval Gates

High-impact actions may require explicit human/user approval after the exact proposed action is known.

Good approval UX shows:
- what action
- target resource
- important parameters
- expected effect

Do not request blanket approval for unknown future actions.

## 13. RAG / Vector Authorization

Retrieval must enforce the same access rules as the source data.

~~~text
query
+ authenticated principal
↓
permission-aware retrieval
↓
authorized chunks only
~~~

Filtering after retrieval can already be too late if sensitive content has reached an untrusted layer.

## 14. Vector / Embedding Weaknesses

Risks include:
- cross-tenant leakage
- poisoned documents
- stale/conflicting data
- weak metadata authorization

Validate documents before indexing and enforce permission-aware retrieval.

## 15. Improper Output Handling

Model output is untrusted data.

Do not directly:
- execute generated shell/code
- interpolate into database queries
- render unsafe markup
- call privileged APIs

without validation/sandboxing appropriate to the application.

## 16. Structured Outputs

If downstream code expects JSON/schema:
1. parse
2. validate schema
3. reject unknown/invalid fields
4. enforce semantic constraints

Schema validity does not imply action authorization.

## 17. Sensitive Information

Controls:
- minimize data exposed to model
- redact/tokenize where appropriate
- authorization before retrieval
- restrict logs
- short retention
- secrets outside prompts

Never assume a hidden prompt keeps a credential secret.

## 18. Unbounded Consumption

AI requests can consume large amounts of:
- tokens
- GPU time
- tool calls
- retrieval queries
- money

Guard with budgets:
- max input/output tokens
- max agent steps
- max tool calls
- request timeout
- user/tenant quotas

## 19. Budget Guard

A budget is a hard deterministic control:

~~~text
used + requested <= limit
~~~

Do not ask the LLM whether it thinks the budget should apply.

## 20. Rate Limits and Backpressure

Safety includes availability.

Use:
- bounded queues
- token/request quotas
- concurrency caps
- circuit breakers

This connects directly to Chapters 63 and 67.

## 21. Data / Model Poisoning

Protect training and retrieval pipelines with:
- source provenance
- validation
- versioning
- access control
- anomaly/review workflows
- reproducible snapshots

Do not automatically train/index arbitrary external content.

## 22. Supply Chain

Track:
- model origin/revision
- package versions
- container digests
- adapters
- datasets
- external tools/plugins

Review updates before production promotion.

## 23. Output Safety Classifiers

A classifier can be one control layer.

Track:
- recall on high-risk classes
- false-positive rate
- calibration
- adversarial robustness

Never assume one classifier catches every unsafe output.

## 24. Threshold Decisions

Generic three-zone policy:

~~~text
score < allow_threshold → allow
middle region → review
score >= block_threshold → block
~~~

Thresholds must be validated for the specific risk and application.

## 25. Safety Evaluation

Create held-out cases for:
- benign requests
- ambiguous cases
- known policy violations
- injection-like untrusted content
- unauthorized retrieval
- excessive tool permissions
- resource-abuse attempts

Evaluate the full system, not only the base model.

## 26. False Positives / False Negatives

False positive:
safe request is unnecessarily blocked.

False negative:
unsafe request passes controls.

Threshold choice is a risk trade-off, not merely accuracy maximization.

## 27. Red-Team Testing

Defensive red-team workflow:
1. define threat class
2. build controlled test cases
3. record expected safe behavior
4. execute in isolated environment
5. fix control failures
6. add regression tests

Keep tests scoped to validating defensive behavior.

## 28. Observability

Log policy metadata such as:
- policy version
- decision
- tool requested
- approval outcome
- budget usage
- error category

Avoid retaining sensitive raw content unnecessarily.

## 29. Incident Response

Prepare:
- disable affected tool/model/version
- rotate compromised credentials
- quarantine poisoned data
- preserve audit metadata
- rollback
- add regression eval

## 30. From Scratch

src/guardrails.py implements:

- risk_priority
- threshold_decision
- budget_allows
- permission_decision
- retrieval_acl_filter
- validate_tool_arguments
- safety_confusion_counts
- safety_metrics

## 31. Common Mistakes

1. system prompt treated as authorization
2. secrets placed in system prompt
3. model output executed directly
4. tool receives broad permanent credentials
5. RAG filters after unauthorized text is exposed
6. unlimited agent/tool/token budget
7. one classifier considered complete security
8. only unsafe cases tested, causing huge false positives
9. safety policy version not logged
10. model-level safety confused with application security

## 32. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 33. Checklist

- [ ] threat model
- [ ] least privilege
- [ ] prompt/content trust boundary
- [ ] tool validation
- [ ] approval gate
- [ ] RAG ACL
- [ ] output validation
- [ ] budgets/rate limits
- [ ] safety evals
- [ ] incident response

## 34. What's Next

Chapter 69 asks a different question: why did the model make a prediction, and how can an explanation be tested for faithfulness?