# Batch 20 Integration Project — Preference Alignment & Evaluation Lab

Batch 20 connects reward modeling, DPO and rigorous evaluation.

## Part A — Reward Model

Synthetic prompts have several candidate responses with hidden utilities.

~~~text
response features
↓
hidden utility
↓
chosen / rejected pairs
↓
linear reward model
↓
Bradley-Terry loss
~~~

Track:
- first/final pairwise loss
- pairwise accuracy
- parameter count

## Part B — DPO

A categorical policy starts equal to a frozen uniform reference.

~~~text
chosen/rejected pairs
+
reference log probabilities
↓
DPO objective
↓
policy logits
~~~

Track:
- first/final DPO loss
- chosen-vs-rejected accuracy
- hidden true-best-response accuracy

This separates training preference accuracy from an independent hidden utility target.

## Part C — Evaluation

The evaluation mode demonstrates:

~~~text
baseline per-sample scores
aligned per-sample scores
↓
paired bootstrap difference

baseline scores
↓
bootstrap confidence interval

training/eval texts
↓
normalized-hash contamination audit
~~~

A reproducibility manifest records seed and bootstrap configuration.

## Run

~~~bash
python integration-project-batch20/src/rlhf_dpo_eval_lab.py --mode reward --steps 40

python integration-project-batch20/src/rlhf_dpo_eval_lab.py \
  --mode dpo \
  --steps 60 \
  --beta 0.5

python integration-project-batch20/src/rlhf_dpo_eval_lab.py --mode evaluate
~~~

## Optional TRL Lab

~~~bash
python -m pip install -r requirements-batch20-trl.txt
~~~

Then reproduce:
- reward modeling with RewardTrainer
- DPO with DPOTrainer / DPOConfig

Record the installed TRL version and configuration because post-training APIs evolve quickly.

## Required Extensions

1. noisy/tied human preferences
2. reward-model held-out prompt split
3. reward calibration
4. reward-hacking synthetic failure case
5. PPO bandit update
6. DPO completion masks
7. reference log-probability cache
8. length-bias audit
9. per-sample JSONL output
10. evaluation slices and judge-order randomization
