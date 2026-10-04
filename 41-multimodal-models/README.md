# Chapter 41 — Multimodal Models

## 1. What Is Multimodal AI?

A multimodal model works with more than one data modality.

Examples:
- text + image
- text + audio
- image + audio
- text + image + audio + video

The main challenge is not simply concatenating inputs; it is learning representations whose information can be aligned and fused.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain modality-specific encoders
- explain shared embedding spaces
- derive CLIP-style contrastive learning
- implement cosine-similarity logits
- explain temperature/logit scale
- compute symmetric image-text contrastive loss
- distinguish early/late/intermediate fusion
- explain cross-attention fusion
- explain image tokens
- understand VLM architecture patterns
- explain frozen vs jointly trained encoders
- understand retrieval vs generation objectives
- identify multimodal evaluation pitfalls

## 3. Two Core Problems

### Alignment

~~~text
image of a cat
↔
"text: a cat"
~~~

Represent corresponding items near each other.

### Fusion

Combine information so one modality can influence processing of another.

## 4. Dual-Encoder Architecture

CLIP-style:

~~~text
image
↓
vision encoder
↓
image embedding
      ↘
        shared space
      ↗
text
↓
text encoder
↓
text embedding
~~~

Each modality has its own encoder.

## 5. Normalize Embeddings

Commonly:

~~~text
z_i =
h_i / ||h_i||
~~~

Then:

~~~text
z_i^T z_j
~~~

is cosine similarity.

## 6. Similarity Matrix

For B aligned image-text pairs:

~~~text
I ∈ R^(B,D)
T ∈ R^(B,D)

S =
I T^T / tau
~~~

Shape:

~~~text
(B,B)
~~~

Diagonal entries are matching pairs.

Off-diagonal entries are in-batch negatives.

## 7. Temperature

Contrastive logits:

~~~text
S_ij =
cos(i,j) / tau
~~~

Smaller tau:
- sharper distribution

Larger tau:
- softer distribution

Some systems learn a logit scale instead of directly learning tau.

## 8. Image-to-Text Loss

For each image row, the correct target is matching text index.

~~~text
L_i2t =
CE(S, arange(B))
~~~

## 9. Text-to-Image Loss

Transpose direction:

~~~text
L_t2i =
CE(S^T, arange(B))
~~~

## 10. Symmetric Contrastive Loss

~~~text
L =
0.5(
L_i2t + L_t2i
)
~~~

This encourages alignment in both retrieval directions.

## 11. In-Batch Negatives

Other examples in the same batch act as negatives.

Larger batches can provide more negatives, but:
- memory cost rises
- false negatives can occur
- duplicated semantic content complicates labels

## 12. Retrieval

After training:

Text-to-image:
1. encode query text
2. normalize
3. compare with image embeddings
4. rank similarities

Image-to-text is symmetric.

## 13. Early Fusion

Combine raw/low-level features early.

Pros:
- rich joint interactions

Cons:
- modalities can have incompatible structure/scales
- expensive

## 14. Late Fusion

Process modalities separately, combine predictions/embeddings late.

Pros:
- modular
- simple

Cons:
- weaker fine-grained interaction

## 15. Intermediate Fusion

Encode modalities first, then allow cross-modal interactions in deeper layers.

This is common in multimodal Transformers.

## 16. Cross-Attention

Example:

~~~text
text hidden states → Q
image tokens       → K,V
~~~

Then text tokens retrieve visual information.

Reverse direction is also possible.

## 17. Image Tokens

Vision encoder may produce patch tokens:

~~~text
image
↓
ViT
↓
(B,N,D_v)
~~~

Project into language-model dimension:

~~~text
(B,N,D_lm)
~~~

Then insert or attend to them as visual tokens.

## 18. Projection Layer

If vision and language dimensions differ:

~~~text
visual features
↓
Linear / MLP projector
↓
LM-compatible features
~~~

The projector can be:
- linear
- MLP
- resampler
- query transformer

## 19. VLM Pattern A — Cross-Attention

~~~text
vision encoder
↓
visual tokens
        ↘
       cross-attention
        ↗
language model
~~~

The LM attends to external vision features.

## 20. VLM Pattern B — Token Injection

Project visual features into LM embedding space and concatenate with text tokens:

~~~text
[visual tokens][text tokens]
↓
decoder-only LM
~~~

Masking determines which positions can interact.

## 21. Frozen Encoders

A practical training strategy:
- freeze vision encoder
- freeze or partially freeze LM
- train projector/adapters

Advantages:
- lower compute
- preserve pretrained capabilities

But capacity may be limited.

## 22. Joint Training

End-to-end training can adapt both modalities deeply.

Costs:
- more compute
- more data
- greater catastrophic-forgetting risk

## 23. Contrastive vs Generative Objectives

Contrastive:
- learn aligned embeddings
- strong retrieval

Generative:
- predict text/tokens conditioned on another modality
- image captioning / VQA / multimodal chat

Many modern systems combine objectives.

## 24. Image Captioning

Conceptually:

~~~text
image encoder
↓
visual features
↓
text decoder
↓
caption tokens
~~~

Train with next-token cross-entropy conditioned on visual input.

## 25. Multimodal Instruction Tuning

Examples contain:
- modality input
- user instruction
- desired response

The model learns task-following behavior across modalities.

This occurs after strong pretrained representations in many systems.

## 26. Missing Modality

Real systems may receive:
- text only
- image only
- both

Architecture/training must define behavior when a modality is absent.

## 27. Modality Dominance

One modality can dominate if it is easier to exploit.

Example:
- text shortcut predicts answer without image

Use ablations:
- image-only
- text-only
- shuffled-image
- shuffled-text

to verify real cross-modal use.

## 28. Evaluation

Retrieval:
- Recall@K
- median rank

Captioning:
- automated text metrics
- human/task evaluation

VQA:
- task accuracy

Multimodal assistants:
- grounding
- hallucination
- robustness
- safety

No single metric covers every failure.

## 29. From Scratch

src/multimodal_numpy.py includes:

- l2_normalize
- similarity_logits
- stable_cross_entropy
- symmetric_contrastive_loss
- retrieval_top1_accuracy
- simple_cross_attention

## 30. Common Mistakes

1. not normalizing before cosine similarity
2. wrong matching diagonal
3. temperature <= 0
4. only training image→text direction unintentionally
5. treating all in-batch examples as true negatives
6. dimension mismatch between vision/LM
7. claiming fusion when one modality is ignored
8. leakage from paired duplicates across splits
9. retrieval evaluation on training pairs
10. interpreting high similarity as proof of grounding

## 31. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 32. Checklist

- [ ] modality encoders
- [ ] shared embedding space
- [ ] contrastive logits
- [ ] temperature
- [ ] symmetric contrastive loss
- [ ] retrieval
- [ ] early/late/intermediate fusion
- [ ] cross-attention
- [ ] visual tokens
- [ ] VLM projector
- [ ] multimodal ablations

## 33. What's Next

Chapter 42 focuses on speech: waveform, STFT, mel features, STT, CTC, seq2seq speech recognition, TTS and vocoders.
