# Chapter 49 Solutions — Key Ideas

- UTF-8 bytes provide a finite 256-symbol base alphabet that can represent arbitrary valid Unicode text.
- BPE greedily merges frequent adjacent token pairs and stores merge order/rank.
- encoding must respect learned merge rank, while decoding concatenates each token's byte sequence.
- deterministic tie-breaking is required for reproducible tokenizer artifacts.
- tokenizer efficiency should be measured across the languages/domains the model will actually serve.
