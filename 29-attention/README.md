# Chapter 29 — Attention

## 1. Why Attention?

RNNs compress past information through a recurrent state.

Attention allows a query position to directly read from many positions:

~~~text
Query
  ↓
compare with Keys
  ↓
attention weights
  ↓
weighted sum of Values
~~~

This creates short paths between distant positions.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain Query / Key / Value
- derive dot-product attention
- explain the sqrt(d_k) scale factor
- implement stable softmax attention
- apply padding masks
- apply causal masks
- explain self-attention vs cross-attention
- split/combine attention heads
- implement multi-head attention from scratch
- understand attention score complexity
- use PyTorch scaled_dot_product_attention
- distinguish masking semantics across APIs
- debug attention shapes

## 3. Query, Key, Value

Given token representations X:

~~~text
Q = X W_Q
K = X W_K
V = X W_V
~~~

Interpretation:

- Query: what information am I looking for?
- Key: what information does this position advertise?
- Value: what information should be retrieved if selected?

These are learned projections, not fixed semantic roles.

## 4. Dot-Product Scores

For one query q and keys k_j:

~~~text
score_j = q · k_j
~~~

For matrices:

~~~text
Scores = Q K^T
~~~

Shapes:

~~~text
Q: (T_q, d_k)
K: (T_k, d_k)

QK^T:
(T_q, T_k)
~~~

## 5. Why Scale by sqrt(d_k)?

If query/key coordinates have roughly unit variance, the dot-product variance grows with d_k.

Large score magnitudes can push softmax into saturation.

Scaled dot-product attention uses:

~~~text
Scores =
QK^T / sqrt(d_k)
~~~

## 6. Softmax

Convert scores into weights:

~~~text
A = softmax(Scores)
~~~

Each query row sums to 1.

Then:

~~~text
Output = A V
~~~

## 7. Stable Softmax

Before exponentiation:

~~~text
scores =
scores - row_max(scores)
~~~

This avoids overflow for large positive values.

## 8. Full Formula

~~~text
Attention(Q,K,V)
=
softmax(
    QK^T / sqrt(d_k)
)
V
~~~

With additive mask M:

~~~text
softmax(
    QK^T / sqrt(d_k)
    + M
)
V
~~~

Masked positions receive a very negative score or are excluded by equivalent API semantics.

## 9. Self-Attention

Q, K, V come from the same sequence:

~~~text
X
├→ W_Q → Q
├→ W_K → K
└→ W_V → V
~~~

Each position can interact with other positions according to the mask.

## 10. Cross-Attention

Queries come from one representation, keys/values from another.

Example:

~~~text
decoder states → Q
encoder states → K,V
~~~

This appears in encoder-decoder Transformers.

## 11. Causal Mask

Autoregressive language modeling must not read future positions.

For T=4:

~~~text
allowed:

q0 → k0
q1 → k0,k1
q2 → k0,k1,k2
q3 → k0,k1,k2,k3
~~~

Matrix:

~~~text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
~~~

The upper triangle is masked.

## 12. Padding Mask

Sequences in a batch can have different valid lengths.

Padding positions should not contribute as keys/values.

A padding mask and a causal mask solve different problems:

- causal: future information
- padding: nonexistent tokens

Sometimes both are needed.

## 13. Mask Semantics Are API-Specific

Some APIs use:
- True = keep
- True = mask
- additive 0 / -inf masks

Never copy a mask from one API to another without reading its semantics.

## 14. Multi-Head Attention

Instead of one attention operation with full dimension, split representation into h heads.

~~~text
d_model = h * d_head
~~~

Each head learns separate projections and attention patterns.

For head i:

~~~text
head_i =
Attention(
    Q_i,
    K_i,
    V_i
)
~~~

Then:

~~~text
Concat(head_1,...,head_h) W_O
~~~

## 15. Shapes

Batch-first self-attention:

~~~text
X:
(B,T,D)

Q/K/V:
(B,T,D)

split heads:
(B,H,T,D_head)

scores:
(B,H,T,T)

weights:
(B,H,T,T)

head output:
(B,H,T,D_head)

combine:
(B,T,D)
~~~

## 16. Why Multiple Heads?

Different heads can learn different projection subspaces and interaction patterns.

However:
- a head is not guaranteed to correspond to one human concept
- visual attention maps alone are not complete explanations of model decisions

## 17. Output Projection

Concatenated heads are mixed by:

~~~text
W_O
~~~

Without this projection, heads would remain independently partitioned at the block output.

## 18. Attention Complexity

For self-attention sequence length T:

~~~text
score matrix size ~ T²
~~~

Compute roughly includes:

~~~text
QK^T: O(T² d)
A V:  O(T² d)
~~~

Memory for dense attention weights can also grow quadratically.

This becomes a core challenge for long context.

## 19. Attention vs RNN

RNN:
- sequential recurrence
- O(T) state transitions
- long gradient path

Self-attention:
- pairwise interactions
- highly parallel over sequence positions
- O(T²) dense score interactions

Neither dominates every latency/memory setting.

## 20. Attention vs Convolution

Convolution:
- local fixed kernel pattern
- strong locality bias

Attention:
- content-dependent interactions
- can directly connect distant positions

Modern architectures sometimes combine both.

## 21. PyTorch SDPA

PyTorch provides:

~~~python
torch.nn.functional.scaled_dot_product_attention(
    query,
    key,
    value,
    ...
)
~~~

Modern PyTorch can dispatch to optimized fused implementations depending on hardware/input/settings.

Use this primitive after understanding the manual formula.

## 22. is_causal

Current SDPA supports causal behavior through an is_causal argument.

Do not combine causal configuration and arbitrary masks without verifying API constraints/semantics for the PyTorch version in use.

## 23. Dropout

Attention dropout is applied to attention probabilities during training in common implementations.

Important:
some low-level APIs require you to explicitly pass dropout_p=0 during evaluation rather than automatically reading module training state.

Read the specific API documentation.

## 24. From Scratch

src/attention_numpy.py includes:

- stable_softmax
- causal_mask
- scaled_dot_product_attention
- split_heads
- combine_heads
- MultiHeadSelfAttentionNumPy

## 25. Numerical Debugging

Inspect:

~~~text
Q/K/V shapes
score min/max
mask
softmax row sums
attention entropy
output shape
NaN/Inf
~~~

If all attention weights are nearly one-hot immediately:
- projections may have large scale
- missing sqrt(d_k)
- initialization may be too large

## 26. Fully Masked Rows

A fully masked query row has no valid key to attend to.

Mathematically, softmax over an empty valid set is undefined.

Production APIs may choose a defined fallback behavior.

Design masks so query rows have intended valid context and test edge cases explicitly.

## 27. Common Mistakes

1. transpose wrong axes
2. divide by sqrt(d_model) instead of sqrt(d_head)
3. mask after softmax
4. causal mask reversed
5. padding mask semantics copied blindly
6. d_model not divisible by heads
7. combine heads without correct transpose
8. softmax across wrong axis
9. interpreting attention weight as causality
10. ignoring T² memory

## 28. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 29. Checklist

- [ ] Q/K/V
- [ ] scaled scores
- [ ] stable softmax
- [ ] causal mask
- [ ] padding mask
- [ ] self vs cross attention
- [ ] multi-head split/combine
- [ ] output projection
- [ ] T² complexity
- [ ] PyTorch SDPA

## 30. What's Next

Chapter 30 wraps attention inside residual, normalization, feed-forward, embedding and positional machinery to build the Transformer.
