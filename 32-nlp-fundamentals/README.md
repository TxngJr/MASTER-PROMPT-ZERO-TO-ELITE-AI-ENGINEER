# Chapter 32 — NLP Fundamentals

## 1. Why NLP Starts Before the Model

A language model never receives “language” directly.

It receives numbers produced by a pipeline:

~~~text
raw text
→ Unicode handling
→ normalization policy
→ tokenization
→ vocabulary / token ids
→ sequences
→ batches / masks
→ model
→ logits
→ language objective
~~~

Bad preprocessing can silently destroy information before the model sees it.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain Unicode/code points
- distinguish normalization from tokenization
- design a vocabulary
- handle unknown tokens
- explain word/character/subword tokenization
- create n-gram counts
- implement a smoothed n-gram language model
- explain conditional language modeling
- calculate cross-entropy and perplexity
- construct next-token training examples
- pad variable-length text
- construct attention/loss masks
- avoid train/test vocabulary leakage
- understand tokenization trade-offs

## 3. Text Is Unicode

Python strings are Unicode text.

Do not assume:
- one character = one byte
- one visible glyph = one code point
- string length equals user-perceived character count

Unicode matters especially for:
- Thai
- emoji
- combining marks
- multilingual corpora

## 4. Unicode Normalization

Unicode can represent visually similar text with different code-point sequences.

Common normalization forms:
- NFC
- NFD
- NFKC
- NFKD

Normalization is a **policy decision**.

Compatibility normalization can change distinctions that may matter in some domains.

## 5. Case Folding

English-centric pipelines often lowercase.

But case can encode meaning:

~~~text
US
vs
us
~~~

Large pretrained models often preserve case depending on tokenizer/model design.

Do not lowercase automatically without task justification.

## 6. Whitespace and Punctuation

Removing punctuation may damage:
- sentiment
- sentence boundaries
- code
- URLs
- named entities

Removing whitespace can be especially harmful in languages where segmentation rules differ.

Text cleaning should preserve information unless there is a documented reason to discard it.

## 7. Tokenization Levels

### Character

~~~text
hello
→ h e l l o
~~~

Pros:
- tiny vocabulary
- no unknown words

Cons:
- long sequences
- weak semantic unit

### Word

~~~text
I love AI
→ I | love | AI
~~~

Pros:
- intuitive units

Cons:
- huge vocabulary
- unknown words
- morphology issues
- segmentation problems

### Subword

~~~text
unbelievable
→ un | believe | able
~~~

Balances vocabulary size and sequence length.

BPE/WordPiece/Unigram receive deeper treatment later.

## 8. Thai Segmentation

Thai does not always separate words with spaces in the same way English does.

A whitespace tokenizer is therefore not a general Thai word tokenizer.

For multilingual/Thai systems:
- use an appropriate tokenizer
- preserve Unicode
- validate segmentation examples manually

## 9. Vocabulary

Map token → integer id.

Example:

~~~text
<PAD> → 0
<UNK> → 1
hello → 2
world → 3
~~~

Keep special-token meanings explicit and stable.

## 10. Vocabulary Leakage

If you build vocabulary from train+validation+test, test information influences representation design.

Safer supervised evaluation:

~~~text
build vocab on training corpus
freeze vocab
map unseen validation/test tokens to UNK
~~~

Large-scale pretrained tokenizers are a different setting because the vocabulary is fixed before downstream evaluation.

## 11. Frequency Threshold

Rare tokens may be replaced by UNK in word-level systems.

Parameter:

~~~text
min_frequency
~~~

Trade-off:
- larger vocab → more sparse rare tokens
- smaller vocab → more UNK

## 12. Sequence Encoding

~~~text
tokens
→ ids
→ [12, 7, 44, 3]
~~~

Decoding uses inverse mapping where available.

Unknown-token handling must be deterministic.

## 13. N-Grams

An n-gram is a sequence of n tokens.

Bigram:

~~~text
(w_(t-1), w_t)
~~~

Trigram:

~~~text
(w_(t-2), w_(t-1), w_t)
~~~

## 14. N-Gram Language Model

Bigram approximation:

~~~text
P(w_t | w_1...w_(t-1))
≈
P(w_t | w_(t-1))
~~~

Maximum-likelihood estimate:

~~~text
P(w_t | w_(t-1))
=
count(w_(t-1),w_t)
/
count(w_(t-1))
~~~

## 15. Zero Counts

Unseen n-grams produce probability 0 under naive maximum likelihood.

Then:

~~~text
log(0)
→ -∞
~~~

Smoothing assigns some probability mass to unseen events.

## 16. Add-k Smoothing

For vocabulary size V:

~~~text
P(w | context)
=
(count(context,w) + k)
/
(count(context) + kV)
~~~

Useful educational baseline, though sophisticated language modeling uses better methods.

## 17. Language Modeling Objective

Autoregressive model factorization:

~~~text
P(x_1,...,x_T)
=
Π_t P(
    x_t | x_<t
)
~~~

Training minimizes negative log-likelihood / cross-entropy over next-token targets.

## 18. Next-Token Pairs

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

This is the same causal objective introduced in Chapter 30.

## 19. Cross-Entropy

For true next token y and predicted distribution p:

~~~text
CE =
-log p(y)
~~~

Average across valid tokens.

For vocabulary logits, use a numerically stable cross-entropy implementation rather than manually computing softmax then log.

## 20. Perplexity

For average natural-log cross-entropy:

~~~text
PPL = exp(CE)
~~~

Interpretation:

lower is better **only when evaluated under compatible tokenization/data/objective**.

Perplexity values across different tokenizers are not directly comparable because token units differ.

## 21. Padding

Variable-length sequences can be padded:

~~~text
A B C PAD PAD
D E F G H
~~~

Padding token must be distinguishable from real tokens.

## 22. Attention Mask

Mask indicates which positions are valid.

Example:

~~~text
1 1 1 0 0
1 1 1 1 1
~~~

Exact semantics depend on API.

## 23. Loss Mask

Do not include padding targets in token loss.

Concept:

~~~text
token_loss * valid_target_mask
~~~

Normalize over valid tokens.

## 24. BOS / EOS

Common special tokens:

- BOS: beginning of sequence
- EOS: end of sequence

EOS lets autoregressive generation learn when to stop.

Not every architecture/tokenizer uses every special token.

## 25. Corpus Splitting

For language modeling, duplicate or near-duplicate text across splits can inflate validation performance.

Important:
- deduplicate where appropriate
- split by document/source boundaries
- avoid future leakage in chronological corpora
- document corpus provenance

## 26. Data Quality

Corpus problems:
- duplicated pages
- broken encodings
- boilerplate
- spam
- personally sensitive data
- train/evaluation contamination
- malformed markup

“More tokens” does not automatically mean better training data.

## 27. Embeddings Preview

Token IDs have no geometric meaning:

~~~text
cat=7
dog=8
~~~

does not mean dog is numerically one unit away from cat.

Embedding tables learn dense vectors:

~~~text
id
→ vector in R^D
~~~

Chapter 33 studies how word vectors can be learned from distributional context.

## 28. From Scratch

src/nlp_basics.py includes:

- Unicode normalization
- simple regex/whitespace-aware word tokenizer
- Vocabulary
- ngram counting
- smoothed NGramLanguageModel
- cross-entropy/perplexity helpers
- next-token window construction

This tokenizer is educational, not a universal multilingual tokenizer.

## 29. Common Mistakes

1. assuming bytes = characters
2. destructive normalization without justification
3. using whitespace-only tokenization for every language
4. building vocabulary on test data
5. no UNK policy
6. including PAD in loss
7. comparing perplexity across different tokenizers blindly
8. data duplicates across splits
9. treating token id values as numeric semantics
10. confusing tokenization with embedding

## 30. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 31. Checklist

- [ ] Unicode
- [ ] normalization
- [ ] tokenization levels
- [ ] vocabulary
- [ ] OOV/UNK
- [ ] n-grams
- [ ] smoothing
- [ ] next-token objective
- [ ] cross-entropy
- [ ] perplexity
- [ ] padding/masks
- [ ] leakage/data quality

## 32. What's Next

Chapter 33 learns dense word representations from distributional context: Word2Vec, GloVe and FastText.
