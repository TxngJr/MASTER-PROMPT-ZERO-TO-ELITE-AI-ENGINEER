# Chapter 36 — Encoder-Decoder Models & T5

## 1. Why Encoder-Decoder?

Some tasks map one sequence into another:

- translation
- summarization
- question answering
- rewriting
- structured generation

Encoder-decoder Transformers separate:

~~~text
source sequence
↓
bidirectional encoder
↓
source representations
        ↓
causal decoder + cross-attention
        ↓
target sequence
~~~

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain encoder-decoder architecture
- distinguish encoder self-attention, decoder self-attention and cross-attention
- construct shifted decoder inputs
- explain teacher forcing
- build source/target padding masks
- explain sequence-to-sequence cross-entropy
- explain T5 text-to-text framing
- construct span-corruption examples
- explain sentinel tokens
- understand relative-position bias conceptually
- compare BERT, GPT and T5
- build a tiny PyTorch encoder-decoder model

## 3. Encoder

The encoder processes the complete source sequence with full self-attention.

~~~text
source ids
↓
embedding
↓
encoder blocks
↓
memory
~~~

Output:

~~~text
(B,S,D)
~~~

where S is source length.

## 4. Decoder

The decoder consumes previous target tokens.

It uses:

1. causal self-attention over target prefix
2. cross-attention over encoder memory
3. feed-forward sublayer

## 5. Decoder Causal Self-Attention

When predicting target token t, the decoder may not read future target tokens.

~~~text
target prefix
y_0 ... y_(t-1)
→ predict y_t
~~~

## 6. Cross-Attention

Queries come from decoder states.

Keys/values come from encoder memory:

~~~text
Q = decoder states
K = encoder memory
V = encoder memory
~~~

This lets each generated target position retrieve relevant source information.

## 7. Cross-Attention Shapes

~~~text
decoder queries:
(B,T,D)

encoder keys/values:
(B,S,D)

scores:
(B,H,T,S)
~~~

Unlike self-attention, query and key sequence lengths can differ.

## 8. Teacher Forcing

Training target:

~~~text
<BOS> A B C <EOS>
~~~

Decoder input:

~~~text
<BOS> A B C
~~~

Prediction target:

~~~text
A B C <EOS>
~~~

The decoder receives ground-truth previous target tokens during training.

## 9. Shift Right

A standard helper creates decoder inputs:

~~~text
labels:
[A,B,C,EOS]

decoder_input:
[BOS,A,B,C]
~~~

Padding/ignored labels must be handled deliberately.

## 10. Sequence Loss

Decoder logits:

~~~text
(B,T,V)
~~~

Targets:

~~~text
(B,T)
~~~

Use token cross-entropy and ignore padded target positions.

## 11. Source Padding Mask

Encoder padding should not affect:
- source self-attention
- decoder cross-attention

## 12. Target Padding Mask

Decoder target padding should not contribute as meaningful keys/values or loss positions.

This is separate from the causal mask.

## 13. Three Mask Types

A seq2seq model may need:

1. source padding mask
2. target padding mask
3. target causal mask

Confusing these is a common bug.

## 14. T5: Text-to-Text

T5 frames many tasks as:

~~~text
text input
→ text output
~~~

Examples conceptually:

~~~text
translate English to Thai: hello
→ สวัสดี
~~~

~~~text
summarize: ...
→ short summary
~~~

Task prefixes can identify the requested transformation.

## 15. Pretraining with Span Corruption

T5 does not simply mask isolated tokens like BERT.

It corrupts spans.

Example:

~~~text
original:
A B C D E F G
~~~

Mask spans C D and F:

~~~text
input:
A B <S0> E <S1> G
~~~

Target:

~~~text
<S0> C D <S1> F <EOS>
~~~

Each removed span is represented by a distinct sentinel token.

## 16. Sentinel Tokens

Sentinels:
- mark removed spans in input
- mark corresponding spans in target
- establish span order

They are special vocabulary tokens.

## 17. Why Span Corruption?

Compared with isolated-token MLM, span corruption trains the model to reconstruct variable-length missing text through an autoregressive decoder.

This directly exercises encoder-decoder generation.

## 18. Corruption Ratio

A pretraining recipe chooses:
- fraction of tokens corrupted
- average span length

Exact settings are training-design choices.

This chapter focuses on the transformation mechanics.

## 19. Relative Position Information

T5-style architectures use relative-position mechanisms rather than relying only on standard learned absolute position embeddings.

The full implementation details are model-family specific.

Later long-context/model-architecture chapters revisit relative position and RoPE systems.

## 20. Text-to-Text Unification

A unified output vocabulary means many tasks can share:
- architecture
- decoder
- loss
- generation code

Only input/output text format changes.

## 21. BERT vs GPT vs T5

### BERT

~~~text
encoder-only
bidirectional
MLM
understanding-oriented
~~~

### GPT

~~~text
decoder-only
causal
next-token LM
generation-oriented
~~~

### T5

~~~text
encoder-decoder
full source + causal target
span corruption / text-to-text
sequence transformation
~~~

## 22. Inference

At inference:
1. encode source once
2. start decoder with BOS/start token
3. generate next token
4. append
5. repeat until EOS/max length

Encoder memory can be reused for every decoding step.

## 23. Decoder KV Cache

Like GPT generation, decoder self-attention can cache past K/V.

Cross-attention can also reuse encoder-side K/V projections because encoder memory is fixed.

Serving optimization chapters revisit this.

## 24. Beam Search Preview

Seq2seq models are often decoded with:
- greedy
- sampling
- beam search

Beam search keeps multiple high-scoring partial hypotheses.

It is not guaranteed to produce the most useful human output.

## 25. From Scratch

src/seq2seq_data.py includes:

- shift_right
- span_corrupt
- make_padding_mask

The span corruption helper explicitly constructs sentinel-ordered source/target sequences.

## 26. Common Mistakes

1. forgetting target causal mask
2. applying causal mask to encoder
3. mixing source and target padding masks
4. off-by-one decoder shift
5. loss on target PAD
6. cross-attention K/V taken from decoder instead of encoder
7. re-encoding source every decoding step unnecessarily
8. sentinel spans overlapping
9. target sentinel order mismatching input
10. claiming T5 is merely BERT plus decoder without objective differences

## 27. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Checklist

- [ ] encoder memory
- [ ] decoder causal self-attention
- [ ] cross-attention
- [ ] shifted targets
- [ ] teacher forcing
- [ ] source/target padding
- [ ] text-to-text
- [ ] span corruption
- [ ] sentinels
- [ ] inference loop
- [ ] BERT vs GPT vs T5

## 29. What's Next

Batch 13 moves beyond standard text Transformers into Graph Neural Networks, Reinforcement Learning and broader Generative AI foundations.
