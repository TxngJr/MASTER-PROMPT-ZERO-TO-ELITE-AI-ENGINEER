# Chapter 70 Exercises

1. Compute sparsity/density for several matrices.
2. Implement global magnitude pruning.
3. Compare layerwise vs global pruning.
4. Remove entire low-norm rows/channels.
5. Measure quality vs sparsity curve.
6. Explain why a dense masked tensor may not save storage.
7. Compute teacher/student soft targets at T=1,2,4.
8. Derive the KD KL objective with T^2 scaling.
9. Compare hard-label-only vs hard+KD student training.
10. Design pruning+quantization evaluation.

Challenge:
- reproduce PyTorch pruning behavior on a tiny network
- distill a larger classifier into a smaller student and benchmark final inference
