# Chapter 59 — DPO & Preference Optimization

## 1. Why DPO?

Classic RLHF can require:
- a reward model
- online generation
- a value model
- PPO optimization
- KL control

Direct Preference Optimization (DPO) learns directly from chosen/rejected pairs using a policy and a reference policy.

~~~text
preference pairs
+
reference model
↓
DPO objective
↓
optimized policy
~~~

## 2. Learning Objectives

You should be able to:

- represent chosen/rejected preference data
- compute sequence log probabilities
- distinguish policy and reference log-probabilities
- derive DPO log-ratio margins
- implement the standard sigmoid DPO loss
- explain beta
- calculate preference accuracy
- understand reference-free variants conceptually
- explain IPO / ORPO / KTO at a high level
- understand length/token masking
- evaluate preference-data quality
- compare DPO with PPO-based RLHF
- recognize overoptimization and distribution shift

## 3. Pairwise Data

Each example contains:

~~~text
prompt x
chosen y_w
rejected y_l
~~~

The preference says:

~~~text
y_w ≻ y_l
~~~

Both responses should be judged under the same prompt/context.

## 4. Sequence Log Probability

For response tokens y_1...y_T:

~~~text
log pi(y|x)
=
Σ_t log pi(y_t | x, y_<t)
~~~

Do not include prompt-only tokens in the completion log-probability comparison unless the formulation explicitly requires it.

## 5. Policy Preference Margin

For the trainable policy:

~~~text
m_policy
=
log pi_theta(y_w|x)
-
log pi_theta(y_l|x)
~~~

Positive means the policy assigns more probability to the chosen response.

## 6. Reference Preference Margin

For frozen reference policy:

~~~text
m_ref
=
log pi_ref(y_w|x)
-
log pi_ref(y_l|x)
~~~

The reference anchors how much the new policy changes preference relative to the starting model.

## 7. DPO Logit

Standard DPO preference logit:

~~~text
z
=
beta * (
m_policy
-
m_ref
)
~~~

Equivalent expanded form:

~~~text
z
=
beta * [
(log pi_theta(y_w)-log pi_ref(y_w))
-
(log pi_theta(y_l)-log pi_ref(y_l))
]
~~~

## 8. DPO Loss

~~~text
L_DPO
=
-log sigmoid(z)
~~~

Stable implementation:

~~~text
softplus(-z)
~~~

A large positive z yields low loss.

## 9. Beta

Beta controls the strength/temperature of the preference-vs-reference trade-off.

Current TRL documents beta as controlling deviation from the reference model.

Interpretation depends on objective conventions, so compare equations rather than relying on a slogan.

## 10. Preference Accuracy

A simple metric:

~~~text
accuracy =
mean(z > 0)
~~~

or equivalently whether the policy-reference preference shift favors chosen over rejected.

Track:
- mean chosen reward/log-ratio
- mean rejected reward/log-ratio
- margin
- accuracy

## 11. Why Reference Policy?

Without the reference term, optimization can reward arbitrary probability changes unrelated to the starting policy.

Reference comparison regularizes preference movement.

## 12. DPO vs Reward Model + PPO

DPO:
- offline chosen/rejected pairs
- no separately trained scalar reward model required
- no online PPO rollout loop required

PPO RLHF:
- explicit reward model
- online policy samples
- value/advantage estimation
- policy-gradient optimization

Neither is universally superior.

## 13. Length Effects

Sequence log-probability sums become more negative with more tokens.

Potential issues:
- response-length bias
- dataset chosen/rejected length imbalance

Audit:
- chosen length distribution
- rejected length distribution
- margin vs length

Do not automatically normalize unless the chosen algorithm specifies it.

## 14. Token Masking

Sequence score should cover the intended completion tokens only.

Mask:
- padding
- prompt prefix when using completion-only likelihood
- invalid/truncated targets

A single off-by-one can invalidate DPO labels.

## 15. Truncation

If chosen/rejected responses are truncated differently, the preference objective can change.

Track:
- prompt truncation
- chosen token count
- rejected token count

Prefer deterministic preprocessing.

## 16. Reference Log-Probability Caching

Reference policy is frozen.

Therefore reference log-probabilities can often be precomputed.

Trade-off:
- extra dataset storage
- faster training
- must invalidate cache when tokenizer/template/reference changes

Current TRL exposes precompute_ref_log_probs in DPO configuration.

## 17. Reference-Free Concept

Some implementations support reference-free modes or alternate objectives.

These alter the regularization assumptions.

Do not silently compare them as if they were standard DPO.

## 18. Label Smoothing / Robust Preference

Preference labels can be noisy.

Robust DPO-style methods may use label smoothing.

Current TRL exposes label_smoothing for compatible DPO variants.

## 19. IPO Concept

Implicit Preference Optimization modifies the loss to avoid some behavior of the logistic DPO objective.

Conceptually it targets a preferred log-ratio gap using a squared objective.

Study the exact paper/objective before implementation.

## 20. ORPO Concept

Odds Ratio Preference Optimization combines:
- supervised likelihood-style learning
- preference odds-ratio optimization

It can avoid a separate reference model in its formulation.

This is not the same equation as standard DPO.

## 21. KTO Concept

KTO can use desirable/undesirable examples without requiring every sample to be an explicit chosen/rejected pair.

It derives from prospect-theory-inspired preference optimization.

## 22. Other Preference Objectives

Modern libraries contain many objectives:
- hinge
- robust
- IPO
- EXO
- NCA
- BCO
- AOT
- APO
- multi-loss combinations

Do not treat "DPOTrainer" as meaning one fixed objective forever.

## 23. Current TRL DPOTrainer

Current TRL supports DPOConfig / DPOTrainer and multiple loss types.

The standard/default preference loss remains sigmoid-style DPO, while modern options include IPO and many other variants.

Always record:
- library version
- loss_type
- beta
- label_smoothing
- reference configuration
- chat template/tokenizer revision

## 24. Preference Dataset Quality

Bad pairs:
- chosen/rejected identical
- trivial formatting differences only
- wrong preference labels
- response lengths systematically biased
- duplicated prompts
- benchmark contamination

Quality audits can outperform blind scaling.

## 25. Preference Strength

Binary chosen/rejected labels hide certainty.

Potential metadata:
- annotator agreement
- score gap
- confidence
- number of votes

Use carefully; not every objective supports weighted preferences directly.

## 26. Evaluation

Before and after DPO evaluate:
- held-out preference accuracy
- task correctness
- instruction following
- formatting
- length
- diversity
- retained capability
- independent human/judge win rates

Do not report DPO training loss alone.

## 27. From Scratch

src/dpo_numpy.py implements:

- preference_margin
- dpo_logits
- dpo_loss
- dpo_pair_accuracy
- chosen_rejected_rewards
- completion_logprob
- length_audit
- ipo_squared_loss

## 28. Common Mistakes

1. policy/reference terms reversed
2. chosen/rejected reversed
3. beta omitted or applied twice
4. prompt tokens accidentally included inconsistently
5. padding contributes to sequence score
6. reference log-prob cache is stale
7. length bias not audited
8. DPO training accuracy mistaken for true preference quality
9. variants compared without reporting loss type
10. held-out preference pairs leak into training

## 29. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 30. Checklist

- [ ] completion log-prob
- [ ] policy margin
- [ ] reference margin
- [ ] DPO logit/loss
- [ ] beta
- [ ] pair accuracy
- [ ] masking
- [ ] length audit
- [ ] reference cache
- [ ] variants
- [ ] held-out evaluation

## 31. What's Next

Chapter 60 builds a rigorous evaluation harness so post-training improvements are measured rather than assumed.
