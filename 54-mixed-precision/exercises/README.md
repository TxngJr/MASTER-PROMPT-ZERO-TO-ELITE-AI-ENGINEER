# Chapter 54 Exercises

1. Compare exponent/fraction bits of FP32/FP16/BF16.
2. Calculate raw storage for one billion parameters in each dtype.
3. Cast a range of values to FP16 and identify overflow/underflow.
4. Simulate BF16 rounding and measure relative error.
5. Scale/unscale tiny gradients.
6. Implement dynamic scale backoff/growth.
7. Explain why clipping must follow unscale.
8. Write a current torch.amp FP16 loop.
9. Write a CPU BF16 autocast smoke test.
10. Compare FP32 and mixed-precision memory/throughput measurements.

Challenge:
- add mixed precision to the Batch 17 tiny LLM trainer
- log scaler, gradient norm and non-finite events
