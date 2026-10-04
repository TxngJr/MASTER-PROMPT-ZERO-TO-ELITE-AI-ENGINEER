# Chapter 22 Exercises

1. Create tensors with float32, float16, int64, bool.
2. Explain view vs reshape on contiguous/non-contiguous tensors.
3. Build a 20→64→32→5 MLP.
4. Count all parameters by hand and verify with code.
5. Write a training loop without helper functions.
6. Demonstrate gradient accumulation when zero_grad is omitted.
7. Compare model.train() and model.eval() using Dropout.
8. Save/load state_dict and verify identical predictions.
9. Benchmark CPU vs CUDA for increasing matrix sizes.
10. Inspect CUDA allocated/reserved memory during a training step.

Challenge:
- write a custom nn.Module implementing a residual MLP block
- write a custom Dataset and collate function
