# Mini Project — Tiny Causal Transformer Language Model

Create a tiny character/token corpus.

Build:
- vocabulary
- token ids
- context windows
- token embedding
- position representation
- 2 causal Transformer blocks
- final normalization
- vocabulary logits
- next-token CrossEntropyLoss

Train a small PyTorch model and generate text with:
- greedy decoding
- temperature sampling
- top-k extension

Track:
- train/validation loss
- perplexity
- parameter count
- tokens/second
- context length
