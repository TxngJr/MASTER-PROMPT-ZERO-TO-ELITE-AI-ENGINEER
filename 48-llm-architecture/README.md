# Chapter 48 — Modern LLM Architecture

## 1. Goal

This chapter assembles the pieces learned in earlier chapters into a modern decoder-only language model.

~~~text
token ids
↓
token embedding
↓
N × decoder blocks
↓
final RMSNorm
↓
LM head
↓
next-token logits
~~~

Each decoder block:

~~~text
x
├─ RMSNorm
├─ Q/K/V projections
├─ RoPE on Q/K
├─ causal attention
├─ output projection
└─ residual add

x
├─ RMSNorm
├─ SwiGLU MLP
└─ residual add
~~~

## 2. Learning Objectives

By the end of this chapter you should be able to:

- trace every tensor shape through a decoder-only LLM
- explain token embeddings and LM heads
- derive RMSNorm
- implement RoPE
- distinguish MHA / MQA / GQA
- implement causal scaled dot-product attention
- explain SwiGLU
- explain pre-norm residual blocks
- estimate parameter counts
- explain weight tying
- explain KV-cache shapes and memory
- distinguish training and autoregressive inference paths
- understand grouped-query attention trade-offs
- build a tiny modern decoder in PyTorch

## 3. Input Tokens

Tokenizer output:

~~~text
input_ids:
(B,T)
~~~

Embedding table:

~~~text
E ∈ R^(V×D)
~~~

Lookup:

~~~text
X = E[input_ids]

X:
(B,T,D)
~~~

## 4. Residual Stream

The model repeatedly transforms a shared hidden representation:

~~~text
x_0
→ x_1
→ x_2
→ ...
→ x_L
~~~

Attention and MLP updates are added to this residual stream.

## 5. Pre-Norm

A common modern structure:

~~~text
x =
x + Attention(Norm(x))

x =
x + MLP(Norm(x))
~~~

Normalization happens before each sublayer.

## 6. RMSNorm

RMSNorm does not subtract the mean.

For vector x ∈ R^D:

~~~text
RMS(x)
=
sqrt(
mean(x_i^2)
+
epsilon
)

RMSNorm(x)
=
gamma ⊙ x / RMS(x)
~~~

PyTorch currently provides nn.RMSNorm directly, but this chapter implements the math first.

## 7. RMSNorm vs LayerNorm

LayerNorm:
- subtract mean
- divide by standard deviation
- affine transform

RMSNorm:
- no mean subtraction
- normalize by root-mean-square magnitude
- learned scale

RMSNorm is common in modern decoder LLMs.

## 8. Attention Projections

For hidden state:

~~~text
X:
(B,T,D)
~~~

Project:

~~~text
Q = X W_Q
K = X W_K
V = X W_V
~~~

Then reshape into heads.

## 9. Multi-Head Attention

If number of heads is H:

~~~text
head_dim =
D / H
~~~

Q/K/V:

~~~text
(B,H,T,head_dim)
~~~

Each head performs attention independently.

## 10. Scaled Dot-Product Attention

~~~text
Attention(Q,K,V)
=
softmax(
Q K^T / sqrt(d)
+
mask
)
V
~~~

Modern PyTorch exposes scaled dot-product attention as a core primitive and may dispatch to optimized kernels depending on device/input.

## 11. Causal Mask

Decoder position t must not read future positions.

For T=4:

~~~text
allowed:

1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
~~~

## 12. Rotary Position Embedding

RoPE rotates pairs of Q/K coordinates using position-dependent angles.

For each pair:

~~~text
[x_even, x_odd]
~~~

apply 2D rotation:

~~~text
x_even' =
x_even cos(theta)
-
x_odd sin(theta)

x_odd' =
x_even sin(theta)
+
x_odd cos(theta)
~~~

The angle frequency depends on dimension index and token position.

## 13. RoPE Frequencies

Typical inverse frequency:

~~~text
inv_freq_i
=
base^(
-2i / d
)
~~~

Angle:

~~~text
theta_(position,i)
=
position * inv_freq_i
~~~

Exact layout/scaling differs across model families.

## 14. Why RoPE?

RoPE injects position into Q/K geometry so attention scores depend on relative positional relationships naturally.

It avoids adding a standalone positional vector to the residual stream.

## 15. MHA vs MQA vs GQA

### MHA

~~~text
H query heads
H key heads
H value heads
~~~

### MQA

~~~text
H query heads
1 key head
1 value head
~~~

### GQA

~~~text
H_q query heads
H_kv key/value heads

H_q % H_kv = 0
~~~

Several query heads share one K/V head.

## 16. Why GQA?

During autoregressive inference, K/V tensors are cached.

Reducing K/V head count reduces:
- KV-cache memory
- memory bandwidth

while retaining multiple query heads.

## 17. GQA Head Expansion

If:

~~~text
H_q = 8
H_kv = 2
~~~

each K/V head serves:

~~~text
8 / 2 = 4
~~~

query heads.

Conceptually repeat/broadcast K/V across query groups for attention math.

## 18. Output Projection

After attention:

~~~text
(B,H,T,d)
↓ transpose/reshape
(B,T,D)
↓ W_O
(B,T,D)
~~~

Then residual addition.

## 19. SwiGLU

Modern LLM MLPs often use gated activations.

A simplified SwiGLU:

~~~text
gate =
SiLU(x W_gate)

value =
x W_up

hidden =
gate ⊙ value

output =
hidden W_down
~~~

where:

~~~text
SiLU(x)
=
x sigmoid(x)
~~~

## 20. Why Gated MLPs?

The gate dynamically controls which transformed features pass through.

SwiGLU-style MLPs are widely used in modern decoder architectures.

## 21. Intermediate Dimension

MLP hidden width is often several times D, but modern gated architectures may use model-specific ratios and rounding.

Do not assume exactly 4D for every model.

## 22. Full Decoder Block

~~~text
input x
│
├── RMSNorm
├── Q/K/V
├── RoPE
├── causal GQA
├── output projection
└── + x
    │
    ├── RMSNorm
    ├── SwiGLU
    ├── down projection
    └── + residual
        ↓
      output
~~~

## 23. Final RMSNorm

After the final block:

~~~text
h =
RMSNorm(x_L)
~~~

then project to vocabulary logits.

## 24. LM Head

~~~text
logits =
h W_vocab^T
~~~

Shape:

~~~text
(B,T,V)
~~~

Training uses next-token cross-entropy.

## 25. Weight Tying

Token embedding and LM-head weights may be shared:

~~~text
W_lm =
E
~~~

Benefits:
- fewer parameters
- shared input/output token geometry

Architecture-specific: not every model ties them.

## 26. Parameter Count

Major contributors:
- token embeddings
- attention projections
- MLP projections
- normalization scales

For GQA:
- Q projection remains H_q heads
- K/V projections shrink with H_kv

## 27. KV Cache

During generation, each layer stores past:

~~~text
K:
(B,H_kv,T,d)

V:
(B,H_kv,T,d)
~~~

For a new token:
- compute only new Q/K/V
- append K/V
- query attends over full cached prefix

## 28. KV Cache Memory

Approximate bytes:

~~~text
2
× layers
× batch
× H_kv
× sequence_length
× head_dim
× bytes_per_element
~~~

Factor 2 is K + V.

## 29. GQA Cache Savings

Compare MHA and GQA with same query-head count:

~~~text
cache ratio
≈
H_kv / H_q
~~~

Example:
- 32 query heads
- 8 KV heads

K/V cache roughly one quarter the head-storage of MHA.

## 30. Training Path

Training:

~~~text
full sequence
↓
parallel causal attention
↓
logits for every position
↓
shifted next-token CE
~~~

No need to cache past tokens because the complete training sequence is processed together.

## 31. Inference Path

Generation:

~~~text
prompt prefill
↓
build KV cache
↓
decode one/few token(s)
↓
append K/V
↓
repeat
~~~

Two phases:
- prefill
- decode

They have different performance bottlenecks.

## 32. Prefill

Prompt length can be large.

Prefill computes:
- all hidden states
- all prompt K/V cache

This phase can use high parallelism.

## 33. Decode

Each step usually processes a very short token batch but reads a growing KV cache.

Decode can become:
- memory-bandwidth bound
- latency sensitive

Serving chapters cover batching/paged attention later.

## 34. Flash Attention / SDPA

Optimized attention implementations reduce intermediate memory traffic and can fuse operations.

The mathematical result remains scaled dot-product attention subject to numerical/kernel details.

Use framework primitives rather than hand-building giant attention-score tensors in production unless needed.

## 35. RoPE and Long Context

Extending context length beyond training range is not automatically safe.

Methods may alter:
- RoPE base/frequencies
- scaling/interpolation
- attention pattern
- training curriculum

Chapter 72 studies long context deeply.

## 36. From Scratch

src/llm_numpy.py includes:

- rms_norm
- rope_angles
- apply_rope
- repeat_kv
- causal_attention
- swiglu
- kv_cache_bytes

## 37. Common Mistakes

1. head_dim not divisible into model_dim
2. H_q not divisible by H_kv
3. RoPE applied to V unintentionally
4. wrong causal-mask direction
5. softmax over wrong attention axis
6. forgetting 1/sqrt(head_dim)
7. repeating Q instead of K/V for GQA
8. cache shape uses H_q instead of H_kv
9. applying final softmax before CE loss
10. assuming training and decode performance are identical

## 38. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 39. Checklist

- [ ] embeddings
- [ ] residual stream
- [ ] RMSNorm
- [ ] RoPE
- [ ] causal SDPA
- [ ] MHA / MQA / GQA
- [ ] SwiGLU
- [ ] final norm
- [ ] LM head
- [ ] weight tying
- [ ] KV cache
- [ ] prefill vs decode

## 40. What's Next

Batch 17 builds the tokenizer and pretraining pipeline required to train a complete LLM from scratch.
