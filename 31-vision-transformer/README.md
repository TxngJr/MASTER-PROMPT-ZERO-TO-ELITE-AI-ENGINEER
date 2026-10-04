# Chapter 31 — Vision Transformer (ViT)

## 1. Why Treat an Image as a Sequence?

A CNN processes local neighborhoods with convolution.

A Vision Transformer converts an image into a sequence of patch tokens:

~~~text
image
↓
split into patches
↓
flatten each patch
↓
linear projection
↓
patch-token sequence
↓
Transformer encoder
↓
classification head
~~~

The core Transformer machinery from Chapters 29–30 can then operate on images.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- derive patch counts
- implement image patchification
- implement linear patch embeddings
- explain CLS token
- explain learned positional embeddings
- build a ViT encoder
- trace all tensor shapes
- calculate major parameter counts
- understand patch-size trade-offs
- compare ViT inductive bias with CNNs
- build a tiny PyTorch ViT
- understand positional interpolation conceptually
- avoid data-layout mistakes

## 3. Image Shape

PyTorch image batch:

~~~text
X:
(B,C,H,W)
~~~

Choose square patch size P.

If H and W are divisible by P:

~~~text
number_of_patches
=
(H/P) * (W/P)
~~~

Each patch contains:

~~~text
C * P * P
~~~

numbers.

## 4. Example

Image:

~~~text
224 × 224 × 3
~~~

Patch:

~~~text
16 × 16
~~~

Patch count:

~~~text
14 × 14
=
196
~~~

Flattened patch dimension:

~~~text
3 * 16 * 16
=
768
~~~

## 5. Patchification

For each non-overlapping patch:

~~~text
(B,C,P,P)
→ flatten
→ (B,C*P*P)
~~~

Across all patches:

~~~text
(B,N,patch_dim)
~~~

where N is number of patches.

## 6. Patch Embedding

Project flattened patch vectors into model dimension D:

~~~text
patch_tokens
=
patches W_patch + b_patch
~~~

Shapes:

~~~text
patches:
(B,N,patch_dim)

W_patch:
(patch_dim,D)

output:
(B,N,D)
~~~

## 7. Conv2D Patch Embedding

A Conv2D with:

~~~text
kernel_size = P
stride = P
out_channels = D
~~~

can implement patch extraction + linear projection efficiently.

PyTorch pattern:

~~~python
nn.Conv2d(
    in_channels=C,
    out_channels=D,
    kernel_size=P,
    stride=P,
)
~~~

Then flatten spatial patch grid into sequence.

## 8. CLS Token

Classification-oriented ViT commonly prepends a learned token:

~~~text
[CLS], patch_1, patch_2, ..., patch_N
~~~

Shape becomes:

~~~text
(B,N+1,D)
~~~

After encoder blocks, the final CLS representation is fed to the classifier.

## 9. Is CLS Required?

No.

Alternatives include:
- mean pooling patch tokens
- attention pooling
- task-specific pooling

CLS is an architectural choice, not a mathematical requirement.

## 10. Positional Embedding

Self-attention needs position information.

Original ViT uses learned position embeddings:

~~~text
position:
(1,N+1,D)
~~~

Add:

~~~text
tokens =
tokens + position
~~~

## 11. Patch Ordering

Patch sequence should use a deterministic spatial ordering, usually row-major:

~~~text
top-left
→ top-right
→ next row
...
~~~

Changing the ordering while keeping old position embeddings changes the model meaning.

## 12. Transformer Encoder

ViT encoder uses full self-attention because the complete image is available.

No causal mask is required for ordinary image classification.

A block contains:
- LayerNorm
- Multi-Head Self-Attention
- residual
- LayerNorm
- MLP/FFN
- residual

## 13. Typical Pre-Norm ViT Block

~~~text
x
├──────────────┐
↓ LayerNorm    │
↓ Attention    │
↓              │
+──────────────┘
↓
x
├──────────────┐
↓ LayerNorm    │
↓ MLP/GELU     │
↓              │
+──────────────┘
↓
output
~~~

## 14. MLP Block

Typical:

~~~text
D
→ expansion
→ GELU
→ dropout
→ D
~~~

Often expansion is around 4D in classic Transformer designs, but it is a tunable architecture choice.

## 15. Patch Size Trade-Off

Small patches:
- more tokens
- finer spatial resolution
- attention cost increases

Large patches:
- fewer tokens
- cheaper attention
- coarse input representation

Because dense attention cost grows roughly with N²:

~~~text
N =
(H/P)*(W/P)
~~~

halving patch size can increase token count dramatically.

## 16. CNN vs ViT Inductive Bias

CNN:
- locality built in
- translation-equivariant convolution
- weight sharing over spatial neighborhoods

ViT:
- weaker locality bias
- global token interactions from early layers
- often benefits strongly from data scale/pretraining/augmentation

On small datasets, a small CNN may be a stronger baseline than a ViT.

## 17. Data Augmentation

Vision Transformers often rely on strong training recipes.

Examples:
- random crop
- flip
- color augmentation
- mixup/cutmix
- label smoothing

Do not attribute performance only to architecture when training recipes differ.

## 18. Position Interpolation

If a ViT pretrained at one image resolution is fine-tuned at another, the patch grid may change.

Learned positional embeddings may therefore need interpolation.

Important:
- CLS position is handled separately
- patch positions must be reshaped into a spatial grid before interpolation

Later transfer-learning chapters revisit this in practice.

## 19. Parameter Count

Patch projection:

~~~text
patch_dim * D + D
~~~

CLS token:

~~~text
D
~~~

Learned position embedding:

~~~text
(N+1)*D
~~~

Each Transformer block contains attention + FFN + norms.

## 20. PyTorch Encoder Layer

Current PyTorch TransformerEncoderLayer supports:

~~~python
nn.TransformerEncoderLayer(
    d_model=D,
    nhead=H,
    dim_feedforward=FF,
    batch_first=True,
    norm_first=True,
    activation="gelu",
)
~~~

batch_first=True keeps:

~~~text
(B,T,D)
~~~

through the encoder API.

## 21. Reference-Layer Caveat

PyTorch documents TransformerEncoderLayer as a reference implementation of the original architecture with fewer features than modern Transformer variants.

That is appropriate for this educational ViT.

Later optimized/model-family chapters use more customized building blocks.

## 22. From Scratch

src/vit_numpy.py includes:

- patch_count
- patchify_nchw
- linear_patch_embedding
- prepend_cls_token
- add_learned_positions

These isolate ViT-specific mechanics before the Transformer encoder.

## 23. Common Mistakes

1. H/W not divisible by patch size
2. mixing NCHW and NHWC
3. wrong patch ordering
4. forgetting CLS increases sequence length by one
5. learned position length mismatch
6. using a causal mask for classification ViT
7. flattening channels in inconsistent order
8. comparing ViT/CNN with very different parameter counts
9. interpreting attention maps as full explanations
10. changing input resolution without handling positions

## 24. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 25. Checklist

- [ ] patch count
- [ ] patchify
- [ ] patch projection
- [ ] CLS token
- [ ] positional embedding
- [ ] encoder
- [ ] no causal mask
- [ ] patch-size trade-off
- [ ] CNN vs ViT bias
- [ ] position interpolation concept

## 26. What's Next

Chapter 32 moves from visual tokens to text: NLP preprocessing, vocabularies, language-model fundamentals and perplexity.
