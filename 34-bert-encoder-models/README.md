# Chapter 34 — BERT & Encoder-Only Models

## 1. Why Encoder-Only Models?

Encoder-only Transformers build a contextual representation for every input token using bidirectional self-attention.

~~~text
tokens
↓
embedding + position
↓
full self-attention encoder blocks
↓
contextual token representations
~~~

They are well suited to understanding-style tasks such as classification, tagging, retrieval embeddings and extractive prediction.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain encoder-only Transformers
- explain bidirectional attention
- distinguish BERT pretraining from GPT pretraining
- construct Masked Language Modeling examples
- explain the original 15% masking policy
- understand CLS / SEP / MASK tokens
- explain token-type/segment embeddings
- explain Next Sentence Prediction historically
- fine-tune an encoder classifier
- mask padding correctly
- understand why MLM is not autoregressive generation
- build a tiny PyTorch BERT-style encoder

## 3. Encoder-Only Architecture

A BERT-style stack:

~~~text
token ids
↓
token embeddings
+ position embeddings
+ optional segment embeddings
↓
Transformer Encoder × N
↓
contextual states
~~~

Self-attention is normally full/bidirectional over valid input positions.

## 4. Bidirectional Context

For token t, representation can attend to tokens on both sides:

~~~text
left context ← token t → right context
~~~

This is powerful for understanding tasks but unsuitable for direct left-to-right generation without changing the masking/objective.

## 5. Masked Language Modeling

Select some tokens and ask the model to reconstruct the originals.

Example:

~~~text
input:
the cat [MASK] home

target at masked position:
went
~~~

Only selected positions contribute to MLM loss.

## 6. Original BERT Corruption Recipe

In original BERT:

- choose 15% of token positions for prediction
- among chosen positions:
  - 80% replace with [MASK]
  - 10% replace with a random token
  - 10% leave unchanged

Labels still contain the original token at all selected positions.

This reduces train/inference mismatch from always seeing [MASK] during pretraining.

## 7. Ignore Labels

Unselected positions should not contribute to MLM loss.

Typical framework convention:

~~~text
label = -100
~~~

for ignored positions when using CrossEntropyLoss with ignore_index=-100.

## 8. Special Tokens

Common BERT-style tokens:

- [PAD]
- [UNK]
- [CLS]
- [SEP]
- [MASK]

Their exact IDs depend on the tokenizer/vocabulary.

## 9. CLS

[CLS] is prepended to the sequence.

The final contextual representation at that position is often used for sequence-level classification.

It is a convention learned during pretraining/fine-tuning, not a mathematically privileged token.

## 10. SEP

[SEP] separates logical segments or marks boundaries.

Example:

~~~text
[CLS] sentence A [SEP] sentence B [SEP]
~~~

## 11. Segment / Token-Type Embeddings

Original BERT adds a learned segment embedding indicating whether a token belongs to segment A or B.

Not every modern encoder uses token-type embeddings.

## 12. Input Representation

Conceptually:

~~~text
x_t =
token_embedding
+ position_embedding
+ segment_embedding
~~~

then LayerNorm/dropout depending on implementation.

## 13. Padding Mask

Padding must not be attended to as meaningful content.

The attention mask marks valid vs padded positions.

Mask semantics differ by API, so inspect the expected convention.

## 14. MLM Loss

Given logits:

~~~text
(B,T,V)
~~~

and labels:

~~~text
(B,T)
~~~

compute cross-entropy only on selected masked-prediction positions.

## 15. Why MLM Is Not Generation

MLM predicts missing tokens using both left and right context.

Autoregressive generation requires:

~~~text
P(x_t | x_<t)
~~~

with no access to future tokens.

Therefore a vanilla BERT MLM head is not a GPT-style generator.

## 16. Next Sentence Prediction

Original BERT also used Next Sentence Prediction (NSP).

Later encoder models often changed or removed NSP.

Treat NSP as a historical BERT design choice, not a universal encoder requirement.

## 17. Fine-Tuning for Classification

~~~text
[CLS] representation
↓
Linear(D → classes)
↓
CrossEntropyLoss
~~~

All encoder parameters may be fine-tuned jointly.

## 18. Token Classification

For NER/tagging:

~~~text
hidden state per token
↓
Linear(D → tag_count)
~~~

Ignore padded positions in the loss.

## 19. Sentence Embeddings Caveat

Raw [CLS] vectors from a generic BERT checkpoint are not automatically optimal semantic-similarity embeddings.

Dedicated sentence-embedding training objectives can improve retrieval/similarity geometry.

## 20. Pooling Options

Possible sequence representations:
- CLS
- mean pooling valid tokens
- max pooling
- learned pooling

Evaluate rather than assuming one is universally best.

## 21. Pretraining vs Fine-Tuning

Pretraining:
- huge generic corpus
- self-supervised objective
- expensive

Fine-tuning:
- smaller labeled task
- initialize from pretrained weights
- adapt parameters/head

Later chapters cover parameter-efficient fine-tuning.

## 22. From Scratch

src/bert_data.py includes:

- add_special_tokens
- create_token_type_ids
- mlm_corrupt
- masked_accuracy

The goal is to make the pretraining data objective explicit before relying on libraries.

## 23. Common Mistakes

1. applying a causal mask to standard BERT MLM
2. computing loss on unselected tokens
3. masking special tokens
4. masking PAD
5. using random replacement from invalid IDs
6. treating MLM as autoregressive generation
7. assuming NSP is mandatory
8. using padded tokens in pooling
9. assuming CLS is always the best sentence embedding
10. leaking validation/test text into task-specific vocabulary building

## 24. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 25. Checklist

- [ ] encoder-only stack
- [ ] bidirectional self-attention
- [ ] MLM
- [ ] 15% / 80-10-10 corruption
- [ ] CLS / SEP / MASK
- [ ] segment embeddings
- [ ] padding mask
- [ ] classification fine-tuning
- [ ] token classification
- [ ] MLM vs autoregressive objective

## 26. What's Next

Chapter 35 studies GPT-style decoder-only causal language models and autoregressive generation.
