# Chapter 35 — GPT & Decoder-Only Models

## 1. Decoder-Only Language Modeling

GPT-style models use stacked causal Transformer blocks:

~~~text
token ids
↓
token + position representation
↓
causal self-attention blocks × N
↓
final normalization
↓
vocabulary logits
~~~

They are trained to predict the next token.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain decoder-only Transformers
- construct causal language-model inputs/targets
- explain teacher forcing for next-token training
- implement causal masks
- explain autoregressive generation
- implement greedy decoding
- implement temperature sampling
- implement top-k sampling
- implement top-p / nucleus filtering
- explain repetition and degeneration
- understand context windows
- explain KV-cache motivation
- distinguish training-time and generation-time computation
- understand weight tying
- build a tiny GPT-style PyTorch model

## 3. Causal Language Model Objective

Factorization:

~~~text
P(x_1,...,x_T)
=
Π_t P(x_t | x_<t)
~~~

Training minimizes the negative log probability of the observed next tokens.

## 4. Shifted Inputs and Targets

Sequence:

~~~text
[A,B,C,D,E]
~~~

Input:

~~~text
[A,B,C,D]
~~~

Target:

~~~text
[B,C,D,E]
~~~

Each position predicts exactly one token ahead.

## 5. Causal Attention

Position t may attend only to positions <= t.

~~~text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
~~~

Future-token leakage invalidates the language-model objective.

## 6. Decoder-Only Block

A modernized educational block:

~~~text
x
↓ LayerNorm
↓ causal self-attention
+ residual
↓ LayerNorm
↓ FFN
+ residual
~~~

This is pre-norm.

Architectural details vary across GPT families.

## 7. Vocabulary Logits

Final hidden states:

~~~text
(B,T,D)
~~~

Projection:

~~~text
D → V
~~~

Logits:

~~~text
(B,T,V)
~~~

Use raw logits with CrossEntropyLoss.

## 8. Weight Tying

A model may reuse the token embedding matrix for the output projection:

~~~text
logits =
h E^T
~~~

This reduces parameter count and couples input/output token geometry.

Not every model family ties weights identically.

## 9. Training Teacher Forcing

During training, each position receives the true previous tokens from the dataset.

The model is not forced to consume its own mistakes during training.

Generation is therefore a different runtime process.

## 10. Autoregressive Generation

~~~text
prompt
↓
forward
↓
last-position logits
↓
choose next token
↓
append token
↓
repeat
~~~

## 11. Greedy Decoding

~~~text
next =
argmax(logits)
~~~

Deterministic.

Pros:
- simple
- reproducible

Cons:
- can become repetitive
- may miss diverse high-probability continuations

## 12. Temperature

Adjust logits:

~~~text
logits_scaled =
logits / temperature
~~~

Temperature < 1:
- sharper distribution
- more deterministic

Temperature > 1:
- flatter distribution
- more randomness

Temperature must be positive.

## 13. Top-k Sampling

Keep only the k highest-scoring tokens.

All others receive probability zero after filtering.

This caps candidate count.

## 14. Top-p / Nucleus Sampling

Sort tokens by probability.

Keep the smallest set whose cumulative probability reaches p.

Candidate count adapts to distribution sharpness.

## 15. Top-k vs Top-p

Top-k:
- fixed number of candidates

Top-p:
- variable candidate set based on probability mass

They can also be combined.

## 16. Numerical Stability

Before softmax:

~~~text
logits -= max(logits)
~~~

Then exponentiate.

Filtering should happen consistently before sampling.

## 17. Repetition

Autoregressive models can fall into repetitive loops.

Possible decoding controls include:
- repetition penalties
- no-repeat n-grams
- frequency/presence penalties
- better sampling strategy

These alter generation behavior and should not be confused with model training quality.

## 18. Context Window

A decoder model can only condition on tokens included in its active context.

If context length is T_max:

~~~text
input length <= T_max
~~~

Longer prompts require:
- truncation
- sliding-window strategies
- architecture/model support for longer context

## 19. Position Handling

GPT-family models may use:
- learned absolute positions
- sinusoidal positions
- RoPE
- other relative schemes

Later long-context chapters cover these deeply.

## 20. KV Cache Motivation

Naive generation recomputes K/V projections for all prior tokens at every new step.

At generation step t:

~~~text
old tokens' K/V
do not change
~~~

Cache them and compute only new-token projections.

## 21. KV Cache Concept

For every layer store:

~~~text
K_cached
V_cached
~~~

New token:

~~~text
q_new
k_new
v_new
~~~

Append k_new/v_new to cache and attend q_new over all cached keys/values.

## 22. Why Cache Does Not Remove All Cost

Attention for a new token still compares against previous keys.

KV cache reduces repeated projection/computation, but:
- memory grows with sequence length
- per-token attention work still grows with context
- cache bandwidth becomes important

Serving chapters revisit this in detail.

## 23. Training vs Cached Inference

Training:
- process many positions in parallel
- full causal attention matrix

Generation:
- one/few new tokens at a time
- reuse past K/V
- sequential token dependency

Optimization priorities differ.

## 24. Stopping

Generation may stop on:
- EOS token
- max new tokens
- task-specific delimiter

Always bound generation loops.

## 25. Perplexity

Causal LM validation metric:

~~~text
PPL = exp(mean token CE)
~~~

Compare only under compatible tokenization/objective/data.

## 26. From Scratch

src/gpt_sampling.py includes:

- make_causal_lm_pairs
- stable_softmax
- apply_temperature
- top_k_filter
- top_p_filter
- sample_from_logits

These isolate generation mechanics from model architecture.

## 27. Common Mistakes

1. future-token leakage
2. softmax before CrossEntropyLoss
3. temperature <= 0
4. applying top-p to unsorted probabilities incorrectly
5. forgetting to renormalize after filtering
6. unbounded generation loop
7. treating greedy output as model's only possible answer
8. assuming KV cache makes attention O(1)
9. comparing perplexity across tokenizers blindly
10. using training dropout during deterministic evaluation unintentionally

## 28. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 29. Checklist

- [ ] decoder-only stack
- [ ] causal mask
- [ ] shifted targets
- [ ] teacher forcing
- [ ] greedy generation
- [ ] temperature
- [ ] top-k
- [ ] top-p
- [ ] context window
- [ ] EOS
- [ ] KV cache concept
- [ ] training vs inference

## 30. What's Next

Chapter 36 combines a bidirectional encoder with a causal decoder and cross-attention: encoder-decoder / T5-style sequence-to-sequence models.
