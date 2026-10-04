# Batch 19 Integration Project — Fine-Tuning, LoRA & Instruction SFT

Batch 19 compares three adaptation paths under one repository.

## Part A — Full Fine-Tuning

A frozen synthetic pretrained linear map is copied into a trainable model.

The target task differs by a known low-rank weight update.

~~~text
base W
↓ full parameter update
target W'
~~~

Track:
- trainable parameters
- initial/final adaptation loss
- retained base-task loss

## Part B — LoRA

Use the same base/target problem.

~~~text
W frozen
+
(alpha/r) B A
↓
adaptation
~~~

The lab verifies:
- base W has requires_grad=False
- B starts at zero
- only A/B are optimized
- trainable fraction is much smaller than full tuning

## Part C — Instruction SFT

Synthetic chat-like token sequences contain:

~~~text
user prompt tokens
↓ ignored labels

assistant response tokens
↓ supervised labels
~~~

A tiny causal GRU language model performs next-token SFT with ignore_index=-100.

This part tests the **data/loss mechanics** of instruction tuning independently from model architecture.

## Run

~~~bash
python integration-project-batch19/src/fine_tune_lora_sft_lab.py --mode full --steps 20
python integration-project-batch19/src/fine_tune_lora_sft_lab.py --mode lora --steps 20
python integration-project-batch19/src/fine_tune_lora_sft_lab.py --mode sft --steps 10
~~~

## Optional PEFT Lab

~~~bash
python -m pip install -r requirements-batch19-peft.txt
~~~

Then reproduce the LoRA experiment using Hugging Face PEFT and a tiny compatible causal-LM checkpoint.

## Required Extensions

1. gradual unfreezing
2. full FT vs LoRA validation curves
3. q/v-only vs all-linear LoRA
4. adapter merge/unmerge in PyTorch
5. quantized-base memory estimate
6. real chat-template preprocessing
7. multi-turn assistant-only labels
8. packed conversations
9. held-out instruction evaluation
10. capability-retention evaluation
