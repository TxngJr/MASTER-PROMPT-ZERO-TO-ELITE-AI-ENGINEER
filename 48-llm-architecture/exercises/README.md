# Chapter 48 Exercises

1. Trace tensor shapes through a decoder block.
2. Implement RMSNorm and compare with LayerNorm.
3. Generate RoPE sin/cos tables.
4. Apply RoPE to Q/K.
5. Convert MHA shapes into GQA shapes.
6. Implement causal attention.
7. Implement SwiGLU.
8. Count parameters for MHA vs GQA blocks.
9. Calculate KV-cache memory for several contexts/dtypes.
10. Implement one tiny decoder block in PyTorch.

Challenge:
- add incremental KV-cache decoding
- verify cached decode logits match full-prefix logits
