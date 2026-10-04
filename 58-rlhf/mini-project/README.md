# Mini Project — Tiny RLHF Bandit

Build a small contextual bandit with several candidate responses per prompt.

1. create hidden ground-truth utilities
2. generate noisy preference pairs
3. train a Bradley-Terry reward model
4. freeze a reference policy
5. optimize a policy with PPO-style clipping + KL shaping
6. compare proxy reward against hidden true utility
7. deliberately create a reward-model blind spot and observe reward hacking

Report both reward-model score and true held-out utility.
