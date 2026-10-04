# Chapter 57 — Instruction Tuning

## 1. What Is Instruction Tuning?

A pretrained base language model learns generic next-token prediction. Instruction tuning teaches the model to follow explicit user requests and conversational structure through supervised fine-tuning.

~~~text
base LM
↓
instruction/chat demonstrations
↓
supervised fine-tuning
↓
instruction-following model
~~~

## 2. Learning Objectives

- distinguish base and instruction-tuned models
- construct instruction/input/response examples
- represent multi-turn conversations
- apply the base model's chat template
- understand role/control tokens
- build full-sequence and assistant-only labels
- preserve causal-shift alignment
- handle EOS and conversation boundaries
- pack conversations safely
- truncate without removing all target tokens
- audit SFT data quality and duplication
- evaluate held-out instruction following
- compare full SFT with LoRA SFT

## 3. Supervised Fine-Tuning

SFT trains on demonstrations of desired behavior.

~~~text
instruction + optional input → desired response
~~~

For a causal LM the formatted conversation becomes one token stream, while labels define which tokens contribute to loss.

## 4. Message Schema

A common intermediate representation:

~~~text
[
  {role: system, content: ...},
  {role: user, content: ...},
  {role: assistant, content: ...}
]
~~~

Keep roles structured until the final chat-template formatting stage.

## 5. Chat Templates

Chat models can expect very different control-token layouts. The template is part of the model interface, much like the tokenizer vocabulary.

Do not invent a new role-token format when adapting an existing chat model unless you deliberately intend to retrain that interface.

## 6. Training-Time Formatting

For an existing Transformers chat model, apply its real template as preprocessing.

~~~python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
~~~

The target assistant response is already present in a training example, so a generation-only prompt marker is not needed.

## 7. Full-Sequence Loss

The simplest causal-LM SFT setup supervises every serialized token:

~~~text
labels = input_ids
~~~

This spends loss on system, user, control and assistant text.

## 8. Assistant-Only Loss

Many recipes supervise only desired assistant content:

~~~text
system tokens     → ignore
user tokens       → ignore
assistant content → supervise
~~~

PyTorch cross-entropy commonly uses ignore index -100.

## 9. Example Label Mask

~~~text
tokens:
[SYS, s1, USER, u1, ASST, a1, a2, EOS]

labels:
[-100, -100, -100, -100, -100, a1, a2, EOS]
~~~

Whether assistant role/control tokens themselves are supervised is a recipe choice; document it.

## 10. Causal Shift

Mask construction and causal shifting are separate concerns.

Framework causal-LM classes may shift labels internally. A custom training loop may need to shift logits/targets manually. Never shift twice.

## 11. Multi-Turn Conversations

Possible supervision policies:
- every assistant turn
- only selected assistant turns
- only final assistant turn

Keep the policy consistent across preprocessing and evaluation.

## 12. System Messages

System messages can define role, task constraints and output format. Contradictory system policies in the SFT corpus create inconsistent targets.

## 13. EOS / End-of-Turn

The model needs explicit conversation boundaries expected by its template. Missing end markers can teach responses to flow into synthetic next roles.

## 14. Packing

Packing several short conversations into one training sequence reduces padding.

Risks:
- label masks can become misaligned
- conversation boundaries can disappear
- attention may cross examples if the model/mask permits it

Insert the correct separators and test masks after packing.

## 15. Truncation

Naive truncation can leave only prompt tokens and no target response.

Track:
- prompt tokens
- supervised response tokens
- total length

Reject examples with zero supervised tokens after truncation.

## 16. Data Quality

Audit:
- correctness
- relevance
- clarity
- contradictions
- formatting artifacts
- exact/near duplicates
- language/domain balance

High-quality demonstrations are often more valuable than simply increasing example count.

## 17. Synthetic Instruction Data

Synthetic data can expand coverage, but can also amplify teacher-model mistakes and stylistic bias. Use filtering and independent held-out evaluation.

## 18. Dataset Mixtures

SFT mixtures may include:
- general instruction following
- coding
- math
- summarization
- multilingual tasks
- domain tasks

Track actual sampled tokens/examples per source rather than only configured weights.

## 19. Full SFT vs LoRA SFT

Full SFT updates the full model and uses more optimizer state. LoRA SFT freezes the base and updates adapters, producing smaller task checkpoints.

Compare them with the same:
- data
- number of updates/tokens
- evaluation protocol

## 20. Evaluation

Training loss is not an instruction-following metric.

Evaluate held-out prompts for:
- correctness
- requested format
- completeness
- instruction adherence
- language/domain behavior
- refusal/insufficient-evidence behavior when relevant

## 21. Exact vs Open-Ended Tasks

Exact tasks can use deterministic metrics. Open-ended tasks may need rubrics, references, pairwise evaluation or human review.

## 22. Contamination

Split by the right unit:
- source document
- prompt family
- conversation identity
- benchmark problem family

Near-duplicate held-out prompts in training invalidate the evaluation.

## 23. SFT Limitations

SFT teaches imitation of demonstrations. It does not directly optimize preferences between competing outputs.

Chapters 58–59 add RLHF and DPO/preference optimization.

## 24. From Scratch

src/instruction_data.py implements:

- validate_messages
- render_simple_chat
- concatenate_role_segments
- assistant_only_labels
- supervised_token_count
- truncate_example
- exact_deduplicate_examples

## 25. Common Mistakes

1. wrong chat template for the base model
2. generation prompt added during training
3. assistant mask shifted incorrectly
4. labels shifted twice
5. truncation removes all supervised tokens
6. packed examples lose boundaries
7. train/eval prompt leakage
8. duplicated low-quality demonstrations dominate
9. only training loss reported
10. full SFT and LoRA compared under different protocols

## 26. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 27. Checklist

- [ ] message schema
- [ ] chat template
- [ ] control tokens
- [ ] assistant-only labels
- [ ] causal shift
- [ ] multi-turn policy
- [ ] EOS / boundaries
- [ ] packing
- [ ] truncation
- [ ] quality / dedup
- [ ] held-out evaluation
- [ ] full vs LoRA SFT

## 28. What's Next

Batch 20 moves from supervised imitation to preference alignment: RLHF, DPO and systematic LLM evaluation.