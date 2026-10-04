# Chapter 30 — Transformer

## 1. Why Transformer?

The Transformer removes recurrence as the primary sequence-processing mechanism.

A basic block combines:

~~~text
token representations
      ↓
self-attention
      ↓
residual + normalization
      ↓
feed-forward network
      ↓
residual + normalization
~~~

Stack many blocks and the model can build increasingly contextual representations.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain token embeddings
- explain positional information
- implement sinusoidal position encoding
- explain residual connections
- implement LayerNorm
- explain Transformer feed-forward networks
- distinguish encoder vs decoder attention
- build a Transformer encoder block
- build a causal decoder block
- explain pre-norm vs post-norm
- calculate parameter counts
- explain next-token language-model training
- use PyTorch Transformer building blocks
- understand why modern production Transformers often use lower-level optimized primitives

## 3. Token Embeddings

Discrete token id:

~~~text
token_id ∈ {0,...,V-1}
~~~

Embedding table:

~~~text
E ∈ R^(V × D)
~~~

Lookup:

~~~text
x_t = E[token_id]
~~~

For a batch:

~~~text
tokens:
(B,T)

embeddings:
(B,T,D)
~~~

## 4. Why Position Is Needed

Self-attention alone compares content.

Without positional information, reordering tokens can leave the set of pairwise content operations unable to distinguish sequence order in the intended way.

We therefore inject position information.

## 5. Sinusoidal Positional Encoding

Original Transformer:

~~~text
PE(pos,2i)
=
sin(
  pos / 10000^(2i/D)
)

PE(pos,2i+1)
=
cos(
  pos / 10000^(2i/D)
)
~~~

Then:

~~~text
X =
token_embedding
+
position_encoding
~~~

## 6. Learned Positional Embeddings

Alternative:

~~~text
position_id
→ learned embedding table
~~~

Pros:
- flexible learned representation

Cons:
- fixed trained position range unless extended carefully

Later LLM chapters cover modern alternatives such as rotary position embeddings.

## 7. Self-Attention Sub-Layer

Input:

~~~text
X ∈ R^(B,T,D)
~~~

Project:

~~~text
Q = XW_Q
K = XW_K
V = XW_V
~~~

Apply multi-head self-attention.

Output remains:

~~~text
(B,T,D)
~~~

so residual addition is shape-compatible.

## 8. Residual Connection

~~~text
y = x + F(x)
~~~

Benefits:
- direct information path
- easier optimization of deep networks
- gradient flow across many blocks

Residual addition requires matching dimensions.

## 9. LayerNorm

For each token vector across feature dimension:

~~~text
mu =
mean(x_features)

variance =
mean((x-mu)^2)

x_hat =
(x-mu) / sqrt(variance + eps)

y =
gamma * x_hat + beta
~~~

LayerNorm does not normalize across the batch like BatchNorm.

## 10. Feed-Forward Network

Position-wise MLP:

~~~text
FFN(x)
=
activation(
    x W1 + b1
)
W2 + b2
~~~

Typical expansion:

~~~text
D
→ D_ff
→ D
~~~

Original Transformer commonly used ReLU.

Modern LLMs often use GELU/SwiGLU-family variants.

## 11. Position-Wise Means Shared Across Time

The same FFN parameters are applied independently to every token position.

~~~text
token 1 ─┐
token 2 ─┼→ same FFN weights
token 3 ─┘
~~~

Attention mixes positions; FFN mixes features within each position.

## 12. Encoder Block

Original post-norm conceptual form:

~~~text
x
↓
Self-Attention
↓
x + attention
↓
LayerNorm
↓
FFN
↓
residual
↓
LayerNorm
~~~

Many modern systems instead use pre-norm.

## 13. Pre-Norm

~~~text
x
↓
LayerNorm
↓
Attention
↓
+ x
↓
LayerNorm
↓
FFN
↓
+ residual
~~~

Pre-norm often improves optimization stability for deep Transformers.

Architecture details vary across model families.

## 14. Encoder

Stack N encoder blocks:

~~~text
embeddings
↓
Encoder Block × N
↓
contextual representations
~~~

Encoder models can use bidirectional/full self-attention when all input tokens are available.

BERT-like models are examples of encoder-focused architectures.

## 15. Decoder Block

Original encoder-decoder Transformer decoder contains:

1. causal self-attention
2. cross-attention to encoder output
3. FFN

Decoder-only language models remove encoder cross-attention and stack causal self-attention blocks.

## 16. Causal Decoder-Only Transformer

~~~text
token ids
↓
embedding + position
↓
causal block
↓
causal block
↓
...
↓
LayerNorm
↓
vocabulary projection
↓
logits for next token
~~~

This architecture is the conceptual foundation of GPT-style language models.

## 17. Vocabulary Projection

Hidden state:

~~~text
H:
(B,T,D)
~~~

Output matrix:

~~~text
W_vocab:
(D,V)
~~~

Logits:

~~~text
(B,T,V)
~~~

Each position predicts a distribution over vocabulary tokens.

## 18. Next-Token Training

Given:

~~~text
tokens:
[a,b,c,d,e]
~~~

Input:

~~~text
[a,b,c,d]
~~~

Targets:

~~~text
[b,c,d,e]
~~~

Causal masking ensures the position predicting c cannot read c or future targets directly.

## 19. Cross-Entropy

Flatten conceptually:

~~~text
logits:
(B*T,V)

targets:
(B*T)
~~~

Use stable CrossEntropyLoss on raw logits.

Do not softmax first.

## 20. Teacher Forcing

During training, the model receives true previous tokens as input.

At generation time, it receives its own previously generated tokens.

This creates exposure differences between training and autoregressive generation.

## 21. Autoregressive Generation

Start with prompt:

~~~text
tokens_1...T
~~~

Loop:

~~~text
forward
→ take last-position logits
→ choose/sample next token
→ append
→ repeat
~~~

Naive generation recomputes previous attention every step.

Later LLM serving chapters introduce KV cache.

## 22. Parameter Count Intuition

Self-attention projections roughly:

~~~text
W_Q,W_K,W_V,W_O
≈ 4D²
~~~

FFN with expansion D_ff:

~~~text
≈ 2 D D_ff
~~~

Embedding:

~~~text
V D
~~~

For many LLMs, block matrices dominate parameters once depth/width is large, while embedding size depends strongly on vocabulary.

## 23. Weight Tying

Input embedding and output vocabulary projection may share weights:

~~~text
W_vocab = E^T
~~~

This reduces parameters and creates a useful representation constraint.

Not all architectures use tying identically.

## 24. Attention Complexity

Per layer dense self-attention includes:

~~~text
O(T² D)
~~~

interaction cost and T² score structure.

FFN roughly scales as:

~~~text
O(T D D_ff)
~~~

Which term dominates depends on T and model width.

## 25. Modern PyTorch Building Blocks

Useful primitives:
- scaled_dot_product_attention
- MultiheadAttention
- TransformerEncoderLayer
- TransformerEncoder
- TransformerDecoderLayer
- TransformerDecoder

PyTorch documentation describes TransformerEncoderLayer as a reference implementation of the original architecture.

For custom modern architectures, lower-level building blocks can provide more flexibility and performance.

## 26. Fused Attention

Current PyTorch SDPA can dispatch among optimized implementations depending on device and inputs.

The conceptual equation remains:

~~~text
softmax(
    QK^T / sqrt(d)
    + mask
)
V
~~~

Never skip understanding the equation just because the kernel is fused.

## 27. Normalization Variants Preview

Modern LLMs may use:
- LayerNorm
- RMSNorm

and different norm placement.

This chapter implements LayerNorm to establish the baseline.

## 28. Positional Variants Preview

Later chapters cover:
- learned absolute positions
- sinusoidal
- relative position methods
- RoPE
- ALiBi

Position handling is central to long-context behavior.

## 29. Transformer Is Not Just Attention

A working Transformer also needs:

- embeddings
- positions
- normalization
- residuals
- FFN
- masking
- output head
- objective
- optimizer
- data pipeline

Attention is one component.

## 30. From Scratch

src/transformer_numpy.py includes:

- sinusoidal_position_encoding
- layer_norm
- gelu
- TransformerBlockNumPy
- causal decoder-style forward pass built on the Chapter 29 attention module concepts

The implementation focuses on forward mechanics rather than training.

## 31. Common Mistakes

1. missing positional information
2. causal mask reversed
3. softmax before CrossEntropyLoss
4. LayerNorm across wrong axes
5. residual shape mismatch
6. d_model not divisible by heads
7. confusing encoder full attention with decoder causal attention
8. using future tokens during language-model training
9. assuming Transformer = attention only
10. comparing architectures with different parameter budgets

## 32. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 33. Checklist

- [ ] embedding
- [ ] position
- [ ] MHA
- [ ] residual
- [ ] LayerNorm
- [ ] FFN
- [ ] encoder
- [ ] decoder
- [ ] causal decoder-only
- [ ] next-token objective
- [ ] vocabulary logits
- [ ] generation loop

## 34. What's Next

Batch 11 moves into Vision Transformers and NLP foundations, then word embeddings.
