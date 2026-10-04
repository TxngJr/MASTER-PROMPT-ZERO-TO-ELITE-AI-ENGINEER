# Chapter 33 — Word2Vec, GloVe & FastText

## 1. Distributional Representations

Classic NLP uses the distributional hypothesis:

> words that occur in similar contexts often acquire similar representations.

Instead of arbitrary token IDs, learn dense vectors:

~~~text
word
→ vector in R^D
~~~

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain distributional semantics
- generate CBOW/Skip-Gram training pairs
- derive the Skip-Gram softmax objective
- explain negative sampling
- implement negative-sampling updates
- explain unigram^0.75 sampling
- build word-context co-occurrence matrices
- derive the GloVe objective
- implement a GloVe training step
- explain FastText character n-grams
- compose OOV word vectors from subwords
- calculate cosine similarity
- evaluate embeddings cautiously

## 3. Token IDs Are Not Embeddings

~~~text
cat → 7
dog → 8
~~~

The fact that the IDs differ by one has no semantic meaning.

Embedding methods learn vectors whose geometry comes from training data and objective.

## 4. CBOW vs Skip-Gram

CBOW:

~~~text
context words
→ center word
~~~

Skip-Gram:

~~~text
center word
→ context words
~~~

## 5. Skip-Gram Pairs

For:

~~~text
the cat sat
~~~

with window size 1:

~~~text
(the, cat)
(cat, the)
(cat, sat)
(sat, cat)
~~~

Each pair is:

~~~text
(center, context)
~~~

## 6. Full Softmax Objective

Center input embedding v_w and output/context embedding u_c:

~~~text
score(w,c)
=
u_c^T v_w
~~~

Probability:

~~~text
P(c|w)
=
exp(u_c^T v_w)
/
Σ_j exp(u_j^T v_w)
~~~

The vocabulary-sized denominator is expensive for large vocabularies.

## 7. Two Embedding Tables

Classic Skip-Gram commonly learns:

- input embeddings
- output/context embeddings

After training, downstream use may choose:
- input embeddings
- output embeddings
- their sum/average

State the choice explicitly.

## 8. Negative Sampling

Replace full softmax with:

- positive observed context
- K sampled negative contexts

Objective:

~~~text
log σ(u_c^T v_w)
+
Σ_k log σ(-u_k^T v_w)
~~~

This turns representation learning into positive-vs-noise discrimination.

## 9. Sampling Distribution

A classic Word2Vec choice:

~~~text
P(w)
∝ count(w)^0.75
~~~

The 0.75 exponent reduces extreme domination by very frequent words.

## 10. Frequent-Word Subsampling

Very common tokens can produce huge numbers of weak training pairs.

Subsampling can probabilistically remove some occurrences to:
- reduce compute
- reduce dominance of frequent function words

## 11. Stable Optimization

For positive score s:

~~~text
loss_positive
=
-log σ(s)
~~~

For negative score n:

~~~text
loss_negative
=
-log σ(-n)
~~~

Use stable sigmoid/softplus-style calculations for extreme scores.

## 12. Cosine Similarity

~~~text
cos(a,b)
=
a·b / (||a|| ||b||)
~~~

It measures angular similarity.

Nearest-neighbor evaluation often uses cosine instead of Euclidean distance.

## 13. Analogy Arithmetic

Exploratory pattern:

~~~text
king - man + woman
≈ queen
~~~

This can emerge in some corpora/models but is not guaranteed and should not be treated as proof of general reasoning.

## 14. GloVe

GloVe learns from a global co-occurrence matrix:

~~~text
X_ij
=
weighted number of times word i
appears near context j
~~~

## 15. GloVe Objective

~~~text
J
=
Σ_ij
f(X_ij)
(
w_i^T wtilde_j
+
b_i
+
btilde_j
-
log X_ij
)^2
~~~

Only positive co-occurrence entries participate.

## 16. Weighting Function

Classic form:

~~~text
f(x)
=
(x/x_max)^alpha   if x < x_max
1                 otherwise
~~~

This avoids letting extremely common co-occurrences dominate the objective.

## 17. Why log(X)?

Co-occurrence counts span large ranges.

Taking:

~~~text
log X_ij
~~~

compresses scale and connects embedding dot products to multiplicative count relationships.

## 18. Word2Vec vs GloVe

Word2Vec:
- prediction/sample based
- local training pairs

GloVe:
- explicit global co-occurrence table
- weighted regression objective

Both implement distributional learning differently.

## 19. FastText

Static word embeddings struggle with unseen words.

FastText represents words using character n-grams.

Example:

~~~text
<where>
~~~

3-grams include:

~~~text
<wh
whe
her
ere
re>
~~~

Exact boundaries/settings depend on the implementation.

## 20. Subword Composition

Conceptually:

~~~text
word_vector
=
word vector
+
Σ character-ngram vectors
~~~

Educational code in this chapter focuses on the subword component directly.

## 21. OOV Behavior

A new word can still receive a vector if some of its character n-grams were seen during training.

This is especially useful for:
- morphology
- spelling variants
- rare word forms

It is not a complete semantic solution.

## 22. Morphology and Language

Subword sharing can help morphologically rich languages.

But character overlap can also connect unrelated words accidentally.

Evaluate downstream usefulness rather than assuming shared substrings imply shared meaning.

## 23. Static Embedding Limitation

One word type usually gets one static vector:

~~~text
bank
~~~

The same vector is used in:
- river bank
- financial bank

Contextual models such as BERT compute token representations conditioned on surrounding words.

## 24. Bias

Embeddings absorb statistical associations from their corpus.

They may encode:
- stereotypes
- demographic bias
- historical imbalance
- domain artifacts

Embedding geometry must be interpreted critically.

## 25. Intrinsic Evaluation

Examples:
- nearest neighbors
- word similarity
- analogies

Useful for fast diagnostics, but may correlate poorly with downstream task quality.

## 26. Extrinsic Evaluation

Use embeddings inside:
- classifier
- retrieval system
- tagging model

Then measure task performance.

This often gives more actionable evidence.

## 27. From Scratch

src/word_embeddings.py includes:

- skipgram_pairs
- negative_sampling_distribution
- SkipGramNegativeSampling
- cooccurrence_matrix
- GloVeModel
- character_ngrams
- FastTextSubwordTable
- cosine_similarity

## 28. Common Mistakes

1. forgetting Skip-Gram often uses two embedding tables
2. sampling the positive context as a negative without intention
3. unstable sigmoid/log math
4. ignoring frequent-token imbalance
5. asymmetric co-occurrence windows by accident
6. log(0) in GloVe
7. missing word-boundary markers in subwords
8. cherry-picking analogy successes
9. treating cosine as semantic truth
10. ignoring corpus bias

## 29. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 30. Checklist

- [ ] CBOW
- [ ] Skip-Gram
- [ ] negative sampling
- [ ] 0.75 distribution
- [ ] cosine similarity
- [ ] co-occurrence matrix
- [ ] GloVe objective
- [ ] character n-grams
- [ ] FastText OOV intuition
- [ ] intrinsic/extrinsic evaluation
- [ ] static contextual limitation
- [ ] embedding bias

## 31. What's Next

Batch 12 introduces contextual pretrained language models: BERT, GPT and encoder-decoder/T5.
