# Chapter 61 Exercises

1. Derive symmetric INT8 scale for a small vector.
2. Derive asymmetric scale + zero point.
3. Compare per-tensor and per-channel error.
4. Implement grouped INT4 quantization.
5. Measure MSE under different group sizes.
6. Clip outliers and compare downstream reconstruction error.
7. Estimate 7B raw weight storage at FP16/INT8/INT4.
8. Explain GPTQ vs naive round-to-nearest.
9. Explain AWQ's activation-aware idea.
10. Inspect a GGUF model's tensor quantization metadata conceptually.

Challenge:
- quantize a tiny PyTorch linear model weight-only and compare outputs
- benchmark an actual supported runtime without changing context/batch settings
