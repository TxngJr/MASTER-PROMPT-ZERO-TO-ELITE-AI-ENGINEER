# Chapter 49 — Tokenizer From Scratch

## 1. Why Tokenization Exists

A language model consumes integer token IDs, not raw strings.

~~~text
raw text
↓
normalization / byte conversion
↓
tokenization
↓
token IDs
↓
embedding lookup
~~~

The tokenizer defines the vocabulary and strongly affects sequence length, multilingual coverage and model efficiency.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- explain Unicode code points and UTF-8 bytes
- distinguish character, word, byte and subword tokenization
- explain vocabulary / IDs / special tokens
- derive Byte Pair Encoding training
- implement BPE merge learning
- encode text with learned merges
- decode token IDs back to bytes/text
- explain byte-level tokenization
- handle invalid UTF-8 decoding policies
- reserve special tokens
- measure compression efficiency
- reason about multilingual tokenization
- detect train/eval leakage in tokenizer training
- serialize tokenizer artifacts reproducibly

## 3. Unicode vs Bytes

Python string:

~~~text
"ภาษาไทย"
~~~

contains Unicode characters.

UTF-8 converts it to bytes:

~~~text
text.encode("utf-8")
~~~

A Unicode character may occupy multiple bytes.

## 4. Why Byte-Level?

A byte vocabulary has exactly 256 possible base byte values.

Benefits:
- every UTF-8 string can be represented
- no unknown character token required
- robust multilingual coverage

Cost:
- raw byte sequences can be long

BPE merges common byte sequences into larger tokens.

## 5. Vocabulary

Tokenizer maps token objects to IDs:

~~~text
token → integer ID
~~~

Base byte tokens:

~~~text
0..255
~~~

Learned merge tokens can receive IDs:

~~~text
256, 257, 258, ...
~~~

Special tokens should use explicitly reserved IDs.

## 6. Byte Pair Encoding

Start with a sequence of base symbols.

Repeatedly:

1. count adjacent pairs
2. choose the most frequent pair
3. create a new token for that pair
4. replace occurrences
5. repeat until vocabulary target / merge budget

## 7. Example

Initial:

~~~text
a b a b a b
~~~

Most frequent pair:

~~~text
(a,b)
~~~

Merge:

~~~text
ab ab ab
~~~

Then a later merge may create:

~~~text
abab
~~~

## 8. BPE Training Corpus

Train tokenizer only on the tokenizer-training corpus.

If tokenizer merge training sees evaluation-only data, it can leak distributional information.

Keep:
- tokenizer training data
- model training data
- validation/test data

versioned and documented.

## 9. Pair Frequency

For token sequence:

~~~text
[1, 2, 1, 2, 3]
~~~

pairs:

~~~text
(1,2)
(2,1)
(1,2)
(2,3)
~~~

Counts:

~~~text
(1,2) → 2
...
~~~

## 10. Deterministic Tie-Breaking

Two pairs can have equal frequency.

Tokenizer training should define deterministic tie-breaking so the same corpus/config yields the same vocabulary.

Example:
- higher count first
- then lexicographic pair order

## 11. Merge Application

Given pair:

~~~text
(A,B) → C
~~~

scan left-to-right.

For:

~~~text
A B A B
~~~

result:

~~~text
C C
~~~

Merges must not overlap inconsistently.

## 12. Merge Rank

Encoding must apply learned merges in training order/rank.

Earlier merges have higher priority.

A convenient representation:

~~~text
merge_ranks[(left,right)] = rank
~~~

## 13. Encoding

Byte-level BPE:

~~~text
text
↓ UTF-8 bytes
↓ base byte IDs
↓ repeatedly apply learned merges
↓ token IDs
~~~

## 14. Decoding

Each token corresponds to a byte sequence.

~~~text
token IDs
↓ token byte sequences
↓ concatenate bytes
↓ UTF-8 decode
↓ text
~~~

A valid tokenizer should round-trip normal text:

~~~text
decode(encode(text)) == text
~~~

## 15. Special Tokens

Examples:
- BOS
- EOS
- PAD
- UNK for non-byte tokenizers
- chat role/control tokens

Special tokens are protocol markers, not ordinary text.

Reserve them deliberately to avoid collisions.

## 16. Special-Token Parsing

Do not blindly interpret arbitrary user text as a control token.

Production tokenizers distinguish:
- ordinary text
- explicitly allowed special-token parsing

This avoids accidental protocol injection at tokenizer boundaries.

## 17. Vocabulary Size Trade-Off

Larger vocabulary:
- shorter sequences
- larger embedding/LM-head matrices
- more rare tokens

Smaller vocabulary:
- longer sequences
- smaller embedding table
- more composition

There is no universal best size.

## 18. Compression Metrics

Useful:

~~~text
tokens_per_character
tokens_per_byte
bytes_per_token
~~~

For byte-level tokenizer:

~~~text
bytes_per_token =
UTF8 byte count / token count
~~~

Higher bytes/token generally means stronger compression, but semantic quality matters too.

## 19. Multilingual Fairness

Tokenizer can encode some languages far less efficiently than others.

Measure per-language:
- tokens per sentence
- bytes per token
- sequence truncation rate

Do not evaluate only English.

## 20. Numbers and Code

Tokenization influences:
- arithmetic strings
- identifiers
- whitespace
- indentation
- punctuation

Inspect representative code/data formats before freezing the tokenizer.

## 21. Vocabulary Artifact

Store:
- version
- base encoding
- special-token mapping
- merge list
- token byte sequences
- normalization policy
- training-corpus hash/config

Tokenizer and model checkpoint must agree.

## 22. Byte-Level BPE From Scratch

src/byte_bpe.py includes:

- bytes_to_base_tokens
- pair_counts
- merge_pair
- train_bpe
- ByteBPETokenizer
- encode
- decode
- bytes_per_token

The implementation is intentionally small and deterministic.

## 23. Common Mistakes

1. confusing Unicode characters with bytes
2. tokenizer trained on test corpus
3. non-deterministic merge ties
4. encode merge order differs from training order
5. decode token table incomplete
6. special-token ID collision
7. using replacement decoding silently during roundtrip tests
8. comparing token counts across incompatible normalization
9. changing tokenizer after model training
10. evaluating efficiency only on English

## 24. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 25. Checklist

- [ ] Unicode / UTF-8
- [ ] bytes
- [ ] vocabulary
- [ ] BPE pair counting
- [ ] deterministic merging
- [ ] merge ranks
- [ ] encode
- [ ] decode
- [ ] special tokens
- [ ] serialization
- [ ] compression evaluation
- [ ] multilingual evaluation

## 26. What's Next

Chapter 50 uses token sequences to pretrain a decoder-only language model from scratch.
