# Mini Project — Tiny LLM Pretrainer

Use the Chapter 49 tokenizer and Chapter 48 decoder architecture.

Build:
- token stream
- train/validation split
- causal windows
- mini-batches
- AdamW
- warmup + cosine decay
- gradient accumulation
- gradient clipping
- validation CE / perplexity
- checkpoint save/resume
- greedy sample generation

First prove the model can overfit a tiny corpus before scaling the dataset.
