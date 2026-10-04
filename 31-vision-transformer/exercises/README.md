# Chapter 31 Exercises

1. Calculate patch counts for 224×224 with P=32/16/8.
2. Calculate patch projection parameters.
3. Implement NHWC patchification.
4. Verify Conv2D patch embedding against explicit patchify+Linear.
5. Build CLS-token mean-pooling alternatives.
6. Trace ViT shapes for B=32,N=196,D=768.
7. Calculate attention-score memory as patch size changes.
8. Compare CNN and ViT parameter budgets.
9. Train a tiny ViT on sklearn digits.
10. Inspect learned positional embeddings.

Challenge:
- implement positional interpolation for a changed patch grid
- implement a ViT encoder using lower-level PyTorch SDPA
