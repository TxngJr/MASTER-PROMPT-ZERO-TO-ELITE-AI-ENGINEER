# Chapter 58 Solutions — Key Ideas

- Bradley-Terry models pairwise preference as sigmoid(reward_chosen - reward_rejected).
- a reward model is a learned proxy and must be evaluated independently for generalization/exploitation.
- exact categorical KL is an expectation over the policy distribution; one sampled log-ratio is only an estimator component.
- PPO clipping constrains probability-ratio movement for each sampled action.
- RLHF success requires independent human/task evaluation, not proxy reward alone.
