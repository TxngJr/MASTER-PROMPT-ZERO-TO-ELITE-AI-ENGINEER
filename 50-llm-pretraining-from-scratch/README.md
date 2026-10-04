# Chapter 50 — LLM Pretraining From Scratch

## 1. Goal

Pretraining teaches a decoder-only language model to predict the next token over a large corpus.

~~~text
token sequence
↓
causal decoder
↓
logits for each position
↓
next-token cross-entropy
↓
optimizer update
~~~

This chapter focuses on the training system, not just the model architecture.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- construct causal LM input/target pairs
- create fixed-length token batches
- calculate cross-entropy and perplexity
- explain AdamW
- implement learning-rate warmup
- implement cosine decay
- explain gradient accumulation
- clip gradient norm
- separate training and validation loops
- save/restore full training state
- reason about checkpoint frequency
- estimate tokens processed
- understand batch size in tokens
- understand scaling trade-offs
- identify training instabilities

## 3. Causal Language Modeling

Given:

~~~text
[t0,t1,t2,t3,t4]
~~~

input:

~~~text
[t0,t1,t2,t3]
~~~

target:

~~~text
[t1,t2,t3,t4]
~~~

At every position the model predicts the next token.

## 4. Loss

For target token y:

~~~text
CE =
-log p(y)
~~~

Average over valid tokens.

Do not apply softmax manually before framework cross-entropy loss; frameworks generally expect raw logits.

## 5. Perplexity

~~~text
PPL =
exp(mean_cross_entropy)
~~~

If loss is very large, exp(loss) can overflow.

Perplexity is meaningful only when tokenizer/evaluation protocol are comparable.

## 6. Training Windows

Long token stream:

~~~text
t0 t1 ... tN
~~~

split into context windows:

~~~text
[x_0 ... x_(T-1)] → [x_1 ... x_T]
~~~

Possible policies:
- contiguous blocks
- random windows
- packed documents
- document-aware boundaries

## 7. Effective Batch Size

If:

~~~text
micro_batch = B
gradient_accumulation = A
devices = D
~~~

then examples/update:

~~~text
B * A * D
~~~

For language models it is often better to think in tokens/update:

~~~text
tokens_per_update =
B * A * D * sequence_length
~~~

## 8. Gradient Accumulation

Instead of one giant batch:

1. run several micro-batches
2. scale loss by accumulation steps
3. accumulate gradients
4. optimizer.step()
5. zero gradients

This reduces activation-memory pressure.

## 9. Correct Loss Scaling

For equal-size micro-batches:

~~~text
(loss / accumulation_steps).backward()
~~~

so accumulated gradient approximates the large-batch mean gradient.

Variable token counts require weighting by valid-token count.

## 10. AdamW

AdamW combines adaptive moments with decoupled weight decay.

Conceptually:
- first moment m
- second moment v
- bias correction
- parameter step
- separate weight decay

Do not implement L2 penalty inside the gradient and call it equivalent to AdamW.

## 11. Weight-Decay Parameter Groups

Commonly exclude from weight decay:
- normalization scales
- biases

Decay large matrix weights such as:
- attention projections
- MLP projections
- embeddings depending on training recipe

Recipes vary; document the choice.

## 12. Learning-Rate Warmup

Early training can be unstable at full LR.

Linear warmup:

~~~text
lr(step)
=
max_lr * step / warmup_steps
~~~

for early updates.

## 13. Cosine Decay

After warmup:

~~~text
progress =
(step-warmup)
/
(total-warmup)

lr =
min_lr
+
0.5(max_lr-min_lr)
[
1 + cos(pi * progress)
]
~~~

## 14. Why Scheduler Step Order Matters

A consistent loop is:

~~~text
backward
↓
clip gradients
↓
optimizer.step
↓
scheduler/update LR for next step
↓
zero_grad
~~~

Document whether your schedule is defined for current or next update.

## 15. Gradient Clipping

Global norm clipping:

~~~text
g ← g *
min(
1,
max_norm / ||g||
)
~~~

helps control unusually large gradient updates.

Clipping every step at an extremely low value can also hide deeper instability.

## 16. Initialization

Initialization influences activation/gradient scale.

Typical transformer training uses carefully scaled random initialization.

Residual branches may use architecture-specific scaling.

Do not blindly apply one initializer to every tensor.

## 17. Mixed Precision Preview

Larger training uses:
- BF16
- FP16
- FP32 master/state variants

This is developed deeply in Chapter 54.

For tiny educational CPU training, FP32 is sufficient.

## 18. Train vs Eval Mode

Validation:

~~~text
model.eval()
inference/no-grad context
~~~

Training:

~~~text
model.train()
~~~

Dropout behavior must match phase.

## 19. Validation Loss

Keep validation data out of optimizer updates.

Evaluate periodically:
- mean validation CE
- perplexity
- token count

Use a fixed validation set/protocol to compare checkpoints.

## 20. Overfitting

If:

~~~text
train loss ↓
validation loss ↑
~~~

the model is fitting training data without generalizing.

For massive pretraining this can occur differently across domains; validation mixtures matter.

## 21. Checkpoints

A resumable checkpoint may include:

~~~text
model.state_dict()
optimizer.state_dict()
scheduler state/config
global_step
tokens_seen
random states
training config
tokenizer version
dataset version
~~~

PyTorch recommends serializing module state dictionaries rather than relying on pickling a full module object for portability.

## 22. Resume Correctness

After resume verify:
- model weights
- optimizer moments
- global step
- learning rate
- dataset position/shuffle state if required
- RNG states

Loading only model weights is fine for inference but is not a full training resume.

## 23. Checkpoint Atomicity

Safer pattern:
1. write temporary checkpoint
2. flush/sync as appropriate
3. rename atomically

This reduces risk of partially written checkpoints.

## 24. Tokens Seen

Track:

~~~text
tokens_seen += valid_target_tokens
~~~

This is more comparable across sequence/batch configurations than epochs alone.

## 25. Throughput

Metrics:
- tokens/second
- samples/second
- step time
- memory
- utilization

A faster step with fewer tokens may not mean better throughput.

## 26. Data / Model Scaling

Training compute roughly grows with:
- parameter count
- training tokens

More parameters with too little data can undertrain.
More data with too small a model may saturate model capacity.

Scaling laws are empirical, not exact universal rules.

## 27. Debugging First 100 Steps

Inspect:
- loss finite?
- gradient norm finite?
- LR correct?
- logits finite?
- labels shifted correctly?
- model beats random baseline?
- validation path works?

Do this before long training.

## 28. Random Baseline

Uniform next-token prediction over V tokens:

~~~text
loss ≈ log(V)
PPL ≈ V
~~~

A working model should eventually improve below this baseline on learnable data.

## 29. From Scratch

src/pretraining_utils.py includes:

- make_causal_windows
- stable_cross_entropy
- perplexity_from_loss
- warmup_cosine_lr
- effective_tokens_per_update
- global_norm
- clip_by_global_norm

## 30. Common Mistakes

1. targets not shifted
2. softmax before CE
3. accumulation loss not scaled
4. optimizer step every micro-batch accidentally
5. scheduler off-by-one
6. validation data enters optimizer
7. no optimizer state in resume checkpoint
8. tokenizer version mismatch
9. reporting epoch instead of tokens seen
10. launching long training before tiny overfit/debug test

## 31. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 32. Checklist

- [ ] causal targets
- [ ] CE / perplexity
- [ ] token windows
- [ ] AdamW
- [ ] warmup
- [ ] cosine decay
- [ ] gradient accumulation
- [ ] clipping
- [ ] validation
- [ ] checkpoint
- [ ] resume
- [ ] tokens/sec / tokens seen

## 33. What's Next

Chapter 51 builds the corpus pipeline that feeds tokenizer training and LLM pretraining reproducibly.
