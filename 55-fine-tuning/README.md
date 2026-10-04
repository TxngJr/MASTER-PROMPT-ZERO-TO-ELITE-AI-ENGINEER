# Chapter 55 — Fine-Tuning

## 1. What Is Fine-Tuning?

Fine-tuning adapts a pretrained model to a narrower task/domain using additional supervised or self-supervised training.

~~~text
pretrained model
↓
task/domain data
↓
fine-tuning
↓
adapted model
~~~

Unlike pretraining, you begin from useful learned representations.

## 2. Learning Objectives

You should be able to:

- distinguish pretraining, continued pretraining and fine-tuning
- perform full-parameter fine-tuning
- freeze/unfreeze selected layers
- build discriminative learning-rate groups
- choose a smaller fine-tuning LR
- understand catastrophic forgetting
- design train/validation/test splits
- avoid data leakage
- use early stopping and checkpoint selection correctly
- compare full fine-tuning with PEFT
- understand domain adaptation
- reason about class imbalance and task-specific heads
- preserve tokenizer/model compatibility

## 3. Pretraining vs Fine-Tuning

Pretraining:
- huge generic corpus
- broad objective
- expensive
- learns reusable representations

Fine-tuning:
- smaller task/domain dataset
- narrower objective
- cheaper
- adapts behavior or representations

## 4. Continued Pretraining

Continued pretraining keeps the language-model objective but changes the data distribution.

Examples:
- legal corpus
- code corpus
- Thai corpus

This differs from supervised fine-tuning, where examples explicitly map inputs/instructions to desired outputs.

## 5. Full Fine-Tuning

All parameters remain trainable:

~~~text
theta' =
theta - eta * grad_theta L
~~~

Pros:
- maximum adaptation capacity

Cons:
- memory expensive
- optimizer state for all parameters
- greater forgetting risk
- one full checkpoint per variant

## 6. Freezing

Frozen parameter:

~~~text
requires_grad = False
~~~

It is excluded from gradient-based updates.

Common strategies:
- train final task head only
- unfreeze top N layers
- gradually unfreeze
- full model

## 7. Trainable Parameter Ratio

~~~text
trainable_ratio =
trainable_parameters
/
total_parameters
~~~

Track this for every experiment.

## 8. Fine-Tuning Learning Rate

Pretrained representations can be damaged by an excessively large LR.

Typical principle:

~~~text
fine_tuning_lr
<
from_scratch_lr
~~~

Exact values depend on:
- model size
- dataset size
- optimizer
- batch size
- parameter subset

## 9. Discriminative Learning Rates

Different parameter groups may use different LRs:

~~~text
bottom pretrained layers → smaller LR
top pretrained layers    → larger LR
new task head            → largest LR
~~~

This is optional, not universally superior.

## 10. Catastrophic Forgetting

Aggressive adaptation can hurt previously learned capabilities.

Symptoms:
- target task improves
- broad validation capability drops

Mitigations:
- lower LR
- fewer steps
- regularization
- replay/mixed data
- freeze layers
- PEFT

## 11. Dataset Protocol

Split before training decisions:

~~~text
train
validation
test
~~~

Use:
- train for gradients
- validation for hyperparameters/checkpoint selection
- test once for final evaluation

## 12. Leakage

Common leakage:
- near duplicates across splits
- same conversation/document split into train and validation
- target labels embedded in features
- benchmark examples included in SFT data

Document-level/group-aware splitting is often required.

## 13. Early Stopping

Monitor validation metric.

Stop when improvement stalls for a configured patience.

Never early-stop on test data.

## 14. Checkpoint Selection

Possible criteria:
- minimum validation loss
- maximum F1
- maximum task accuracy
- domain-specific score

Store the criterion with the checkpoint.

## 15. Task Heads

A pretrained backbone may feed:
- classification head
- regression head
- token classifier
- causal LM head
- sequence-to-sequence head

The loss must match the task.

## 16. Class Imbalance

For classification:
- stratified splits
- class weighting
- threshold tuning
- precision/recall/F1

Accuracy alone can be misleading.

## 17. Regularization

Useful options:
- weight decay
- dropout
- smaller LR
- data augmentation
- early stopping
- label smoothing when appropriate

## 18. Domain Shift

Fine-tuning data should represent deployment conditions.

A model fine-tuned on clean benchmark text can fail on:
- noisy user input
- different language variety
- longer context
- different class prevalence

## 19. Evaluation Before and After

Always record baseline before adaptation.

~~~text
base model metric
vs
fine-tuned model metric
~~~

Also monitor retained capabilities where relevant.

## 20. From Scratch

src/fine_tuning_utils.py includes:

- parameter_partition
- trainable_ratio
- freeze_by_prefix
- unfreeze_by_prefix
- discriminative_lr_groups
- early_stopping_update
- select_best_checkpoint

## 21. Common Mistakes

1. no baseline before fine-tuning
2. test set used for checkpoint selection
3. LR copied from pretraining
4. tokenizer/model mismatch
5. accidental full fine-tuning when only head intended
6. forgetting to set train/eval modes
7. duplicate documents across splits
8. class imbalance ignored
9. overfitting tiny dataset
10. saving only weights without experiment config

## 22. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 23. Checklist

- [ ] full fine-tuning
- [ ] freezing/unfreezing
- [ ] trainable ratio
- [ ] LR groups
- [ ] validation protocol
- [ ] early stopping
- [ ] checkpoint selection
- [ ] catastrophic forgetting
- [ ] domain shift
- [ ] baseline comparison

## 24. What's Next

Chapter 56 reduces adaptation cost using LoRA, QLoRA and other parameter-efficient fine-tuning methods.
