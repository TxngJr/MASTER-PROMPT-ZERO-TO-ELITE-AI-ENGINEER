# WordPiece and Unigram Tokenization

Chapter 49 implements byte-level BPE as the primary from-scratch tokenizer. This addendum completes the other major subword families.

## WordPiece

WordPiece begins with an initial vocabulary and repeatedly adds useful subword units according to a likelihood/frequency-oriented scoring rule. In practical implementations, tokenization commonly uses greedy longest-match-first segmentation from the learned vocabulary.

Example vocabulary:
play, ##ing, player

The continuation marker is a convention used by common implementations; the key idea is that words can be decomposed into reusable subword units.

### Build outline
1. normalize/split training text according to the chosen pre-tokenization rules
2. initialize base symbols
3. count candidate merges/subword statistics
4. score candidates
5. add the best candidate
6. repeat until the vocabulary budget is reached
7. tokenize new text with deterministic segmentation and an unknown-token policy

## Unigram language-model tokenizer

Unigram starts with an over-complete candidate vocabulary. It assigns a probability to each token and chooses segmentations maximizing sequence likelihood.

For segmentation x = t1...tk:

log P(x) = sum_i log P(t_i)

Training repeatedly:
1. estimates token probabilities
2. evaluates the loss increase caused by removing candidates
3. prunes low-value tokens
4. re-estimates probabilities
until the target vocabulary size is reached.

Dynamic programming is used to find high-probability segmentations.

## Comparison
- BPE grows vocabulary by merges.
- WordPiece grows useful units and commonly uses greedy segmentation.
- Unigram begins large and prunes while using a probabilistic segmentation objective.

## Data boundary
Fit tokenizer vocabulary only on the training split. Freeze it before validation/test tokenization to avoid vocabulary leakage.

## Exercise
Implement longest-match WordPiece decoding for a tiny fixed vocabulary, then implement dynamic programming for the best Unigram segmentation from manually supplied token log-probabilities.
