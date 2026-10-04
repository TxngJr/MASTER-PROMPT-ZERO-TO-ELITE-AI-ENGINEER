# Chapter 58 — RLHF

## 1. What Is RLHF?

Reinforcement Learning from Human Feedback adapts a language model toward human preferences.

A classical pipeline is:

~~~text
pretrained model
↓
supervised instruction tuning
↓
preference comparisons
↓
reward model
↓
policy optimization
↓
aligned policy
~~~

The difficult part is not merely "run PPO"; it is building a reliable preference signal and preventing the policy from exploiting it.

## 2. Learning Objectives

You should be able to:

- represent preference-pair datasets
- derive the Bradley-Terry reward-model objective
- train/evaluate pairwise reward models
- distinguish reward accuracy from reward calibration
- explain policy/reference models
- compute exact categorical KL
- understand sampled log-ratio KL estimators
- derive PPO probability ratios
- explain clipped PPO objectives
- normalize advantages
- understand value models / advantages
- explain KL penalties
- identify reward hacking
- reason about overoptimization
- design offline evaluation before online optimization
- understand modern TRL reward/PPO tooling conceptually

## 3. Preference Data

A pair contains:

~~~text
prompt x
chosen response y_w
rejected response y_l
~~~

Humans/annotators indicate:

~~~text
y_w ≻ y_l
~~~

Preference data can come from:
- human ranking
- pairwise annotation
- carefully controlled synthetic preference labels

Label quality matters more than raw pair count.

## 4. Reward Model

The reward model maps prompt+response to a scalar:

~~~text
r_phi(x,y) ∈ R
~~~

Higher score should correspond to more preferred responses.

## 5. Bradley-Terry Model

For chosen/rejected rewards:

~~~text
P(y_w ≻ y_l)
=
sigmoid(
r_w - r_l
)
~~~

The pairwise negative log-likelihood is:

~~~text
L_RM
=
-log sigmoid(r_w - r_l)
~~~

Equivalent stable implementation:

~~~text
softplus(-(r_w-r_l))
~~~

## 6. Reward Margin

~~~text
margin =
r_chosen - r_rejected
~~~

Positive margin means the reward model ranks the pair correctly.

Pairwise accuracy:

~~~text
mean(margin > 0)
~~~

Accuracy alone does not measure reward calibration or scale.

## 7. Preference Ties / Noise

Human preferences can be:
- ambiguous
- inconsistent
- annotator-dependent

Possible practices:
- allow ties/uncertainty
- majority vote
- adjudication
- annotator-quality analysis

Do not pretend pair labels are perfect ground truth.

## 8. Reward Model Generalization

A reward model can overfit training comparison styles.

Hold out:
- prompt families
- domains
- lengths
- adversarial examples

Evaluate OOD preference accuracy when deployment prompts differ.

## 9. Policy

The policy is the language model being optimized:

~~~text
pi_theta(y|x)
~~~

It generates candidate responses.

## 10. Reference Policy

A frozen reference policy:

~~~text
pi_ref
~~~

anchors optimization near a known baseline.

Without regularization, policy optimization may move aggressively toward reward-model quirks.

## 11. Exact Categorical KL

For distributions p and q over the same support:

~~~text
KL(p || q)
=
Σ_i p_i log(p_i/q_i)
~~~

It is always non-negative.

For full language-model sequences this exact quantity can be expensive.

## 12. Sampled Log-Ratio

For a sampled token/action a:

~~~text
log pi_theta(a)
-
log pi_ref(a)
~~~

This quantity is often used in policy-training estimators/penalties.

Important:
- a single sampled log-ratio can be negative
- it is not itself the exact full-distribution KL

## 13. KL-Shaped Reward

A common conceptual form:

~~~text
reward_shaped
=
reward_model_score
-
beta * sampled_log_ratio
~~~

Beta controls regularization strength.

Large beta:
- stays closer to reference

Small beta:
- permits more movement

## 14. PPO Ratio

Given old policy and new policy:

~~~text
ratio_t
=
exp(
log pi_new(a_t)
-
log pi_old(a_t)
)
~~~

Ratio > 1 means the new policy increased probability of that sampled action.

## 15. Advantages

Advantage estimates whether an action performed better than the value baseline.

~~~text
A_t ≈ return_t - V(s_t)
~~~

Modern PPO pipelines may use generalized advantage estimation, value models and whitening/normalization.

## 16. Advantage Normalization

Simple normalization:

~~~text
A_hat =
(A - mean(A))
/
(std(A) + epsilon)
~~~

This changes optimization scale, not preference labels.

## 17. PPO Clipping

Clipped surrogate:

~~~text
min(
ratio * A,
clip(ratio, 1-eps, 1+eps) * A
)
~~~

Policy optimization maximizes this surrogate.

Clipping limits overly large probability-ratio updates.

## 18. Value Model

A value model estimates expected future reward.

In language-model PPO it helps estimate advantages.

Poor value estimates can increase policy-gradient variance.

## 19. RLHF Optimization Loop

~~~text
prompts
↓
policy generates responses
↓
reward model scores responses
↓
reference log probabilities
↓
advantages / returns
↓
PPO update
↓
repeat
~~~

This is online relative to policy generations.

## 20. Reward Hacking

The policy optimizes the learned reward, not human intent directly.

It can discover:
- stylistic shortcuts
- excessive verbosity
- repeated phrases
- hidden reward-model biases

High reward-model score does not guarantee true quality.

## 21. Overoptimization

As policy optimization continues:
- proxy reward may keep rising
- true human preference can flatten or fall

Therefore monitor independent evaluation, not reward alone.

## 22. KL Collapse / Excessive Drift

If policy drifts too far:
- language quality can degrade
- reward exploitation can increase
- retained capabilities can fall

Track:
- approximate KL
- response entropy
- task metrics
- independent preference win rate

## 23. Reward Scaling

Reward magnitude affects optimization.

Possible operations:
- centering
- normalization
- clipping

Do not change reward scale without understanding its effect on PPO/value learning.

## 24. Current TRL Concepts

Current Hugging Face TRL includes:
- RewardTrainer for reward models
- DPOTrainer and other offline preference methods
- PPO functionality under its experimental PPO API

Modern post-training ecosystems increasingly use offline preference optimization where suitable because it is simpler than a full online PPO stack.

## 25. From Scratch

src/rlhf_numpy.py implements:

- sigmoid_stable
- bradley_terry_probability
- reward_model_loss
- pairwise_accuracy
- categorical_kl
- sampled_log_ratio
- kl_shaped_reward
- normalize_advantages
- ppo_ratio
- ppo_clipped_policy_loss

## 26. Common Mistakes

1. reward-model training pairs leak into evaluation
2. reward score treated as ground-truth human utility
3. pairwise accuracy confused with calibration
4. sampled log-ratio called exact KL
5. PPO old/new policy probabilities mixed up
6. wrong sign on advantage objective
7. policy updated without a reference/KL monitor
8. reward-model exploitation ignored
9. only proxy reward reported
10. online RL started before offline pipeline is validated

## 27. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Checklist

- [ ] preference pairs
- [ ] Bradley-Terry
- [ ] reward model
- [ ] reference policy
- [ ] exact KL
- [ ] sampled log-ratio
- [ ] PPO ratio
- [ ] advantages
- [ ] clipping
- [ ] value model
- [ ] reward hacking
- [ ] independent evaluation

## 29. What's Next

Chapter 59 derives Direct Preference Optimization, which learns directly from chosen/rejected pairs without running a separate online PPO loop.
