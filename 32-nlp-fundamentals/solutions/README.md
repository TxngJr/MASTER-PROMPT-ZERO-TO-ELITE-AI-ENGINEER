# Chapter 32 Solutions — Key Ideas

- Unicode code points, bytes and visible glyphs are different concepts.
- normalization and tokenization are separate policies.
- vocabulary should normally be fit without peeking at held-out evaluation data.
- token ids are categorical identifiers, not scalar semantic values.
- language-model perplexity is exp(mean cross-entropy) under a compatible tokenization/objective.
- padding positions should not contribute to token prediction loss.
