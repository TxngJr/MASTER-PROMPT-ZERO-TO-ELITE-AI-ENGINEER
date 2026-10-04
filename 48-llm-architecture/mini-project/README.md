# Mini Project — Tiny Modern Decoder

Build a decoder-only language model with:
- token embeddings
- RMSNorm
- RoPE
- grouped-query attention
- causal SDPA
- SwiGLU
- residual connections
- final RMSNorm
- vocabulary head

Train on a tiny internal token corpus.

Then implement:
- prompt prefill
- KV cache
- one-token decode
- greedy generation

Verify cached and uncached next-token logits numerically.
