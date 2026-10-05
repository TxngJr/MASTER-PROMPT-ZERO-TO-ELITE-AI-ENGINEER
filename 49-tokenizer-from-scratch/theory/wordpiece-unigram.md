# WordPiece and Unigram Tokenization

Chapter 49 implements byte-level BPE. This supplement completes the comparison with two other major subword families.

## WordPiece

WordPiece starts from a base vocabulary and iteratively chooses subword additions/merges using a score intended to favor useful combinations rather than pure pair frequency alone.

A commonly taught intuition compares pair frequency with component frequencies:

~~~text
score(a,b) ∝ freq(ab) / (freq(a) * freq(b))
~~~

Exact training details vary by implementation, so treat the formula as intuition unless reproducing a specified implementation.

At encoding time, WordPiece commonly uses greedy longest-match-first segmentation over the learned vocabulary and marks continuation pieces according to tokenizer convention.

### Failure to test

If a word cannot be segmented into legal pieces under a tokenizer's fallback policy, encoding must have a defined behavior (unknown token, bytes/characters, etc.). Never leave it implicit.

## Unigram language-model tokenizer

Unigram starts with an over-complete candidate vocabulary and assigns token probabilities. A segmentation (s=(t_1,...,t_m)) has probability:

~~~text
P(s) = product_i P(t_i)

log P(s) = sum_i log P(t_i)
~~~

Encoding chooses a high-probability segmentation, often solved with dynamic programming/Viterbi-style search. Training repeatedly removes tokens whose deletion least harms the corpus likelihood until the target vocabulary size is reached.

## BPE vs WordPiece vs Unigram

| Family | Training idea | Encoding idea |
|---|---|---|
| BPE | repeated pair merges | apply learned merges |
| WordPiece | add useful subwords | greedy longest match |
| Unigram | probabilistic token inventory + pruning | best-probability segmentation |

## Invariants

- deterministic encode for fixed model/settings;
- decode round-trip for representable text under the defined fallback;
- tokenizer trained on training split only;
- special IDs do not collide with learned vocabulary;
- serialization/deserialization preserves identical token IDs.
