# Chapter 29 Exercises

1. Calculate QK^T by hand for three tokens.
2. Explain the sqrt(d_k) scale using variance intuition.
3. Implement padding masks.
4. Implement causal masks for rectangular query/key lengths.
5. Trace all multi-head shapes for B=8,T=128,D=512,H=8.
6. Compare one head vs multiple heads with equal total D.
7. Measure attention entropy.
8. Compare manual attention against PyTorch SDPA on CPU.
9. Visualize causal attention weights.
10. Benchmark T=64,128,256,512 memory/time.

Challenge:
- implement cross-attention
- implement grouped-query attention shape transformations
