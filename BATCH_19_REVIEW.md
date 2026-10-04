# Batch 19 Review — Chapters 55–57

## Chapters

- 55 — Fine-Tuning
- 56 — LoRA / QLoRA / PEFT
- 57 — Instruction Tuning

## Fine-Tuning Skills

- full-parameter adaptation
- freezing / unfreezing
- trainable-parameter accounting
- discriminative learning rates
- validation/test protocol
- early stopping
- checkpoint selection
- domain shift
- catastrophic forgetting
- retained-capability evaluation

## LoRA / QLoRA / PEFT Skills

- low-rank update BA
- rank / alpha / dropout
- zero-update initialization
- target modules
- trainable parameter count
- merge / unmerge
- adapter checkpoints
- multi-adapter operation
- QLoRA frozen quantized base
- NF4 / double-quantization concepts
- generic codebook quantization vs real NF4 distinction
- current Hugging Face PEFT configuration concepts

## Instruction-Tuning Skills

- instruction / chat schemas
- chat templates
- role/control tokens
- full-sequence vs assistant-only loss
- ignore-index masking
- causal-shift alignment
- multi-turn supervision
- EOS / boundaries
- packing
- truncation
- SFT data quality / dedup
- held-out instruction evaluation
- full SFT vs LoRA SFT

## Implemented From Scratch

### Chapter 55
- parameter_partition
- trainable_ratio
- freeze_by_prefix
- unfreeze_by_prefix
- discriminative_lr_groups
- EarlyStoppingState
- early_stopping_update
- select_best_checkpoint

### Chapter 56
- lora_parameter_count
- lora_delta
- lora_forward
- merge_lora
- unmerge_lora
- trainable_fraction
- nearest_codebook_quantize

### Chapter 57
- validate_messages
- render_simple_chat
- concatenate_role_segments
- assistant_only_labels
- supervised_token_count
- truncate_example
- exact_deduplicate_examples

## Integration Project

Three modes:

1. full fine-tuning on a controlled low-rank adaptation target
2. LoRA tuning on the same base/target mapping
3. assistant-only causal SFT on synthetic chat sequences

## Current API Audit

### Hugging Face PEFT

Current PEFT continues to use:
- LoraConfig
- get_peft_model / adapter attachment
- explicit or default target modules
- target_modules="all-linear" for QLoRA-style broad linear targeting

Adapter checkpoints require compatible base-model architecture/revision.

### Transformers Chat Templates

Current Transformers guidance treats chat templates as part of tokenizer/model formatting.

For training:
- preprocess conversations with the model's template
- use add_generation_prompt=False
- preserve the actual assistant response in the example

## Methodology Audit

### Fine-Tuning
- baseline exists before adaptation
- full FT and LoRA use the same synthetic adaptation target
- retained base-task loss exposes forgetting
- validation/test responsibilities are documented

### LoRA
- base weight is frozen in the PyTorch integration
- B starts at zero so initial adapter output is zero
- only A/B are optimized
- trainable fraction is measured
- generic quantization helper is explicitly not presented as NF4

### Instruction SFT
- prompt tokens use ignore_index=-100
- assistant response tokens are supervised
- causal shift occurs exactly once in the custom training loop
- examples retain explicit boundaries
- supervised-token fraction is reported

## Interpretation Audit

- lower trainable parameter count does not guarantee equal task quality
- LoRA rank is a capacity hyperparameter
- QLoRA saves frozen-base storage/state but activations still matter
- 4-bit storage does not mean all arithmetic is 4-bit
- training loss is not instruction-following evaluation
- chat-template mismatch can invalidate an otherwise correct SFT run
- lower target-task loss can coincide with worse retained capability

## Exit Gate

Before Chapter 58:

1. Batch 19 Core CI passes
2. Batch 19 PyTorch smoke passes
3. run full FT and LoRA on the same adaptation task
4. calculate trainable parameter ratios
5. derive LoRA BA and alpha/r scaling
6. explain why zero-initialized B is a no-op
7. explain QLoRA without claiming direct 4-bit base updates
8. build assistant-only labels
9. verify truncation never leaves zero supervised tokens
10. evaluate held-out instruction behavior rather than training loss alone
