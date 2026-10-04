# Chapter 78 Exercises

1. Calculate FP32/FP16/INT8 storage for the same tensors.
2. Build a model + activation + arena RAM budget.
3. Apply a 10–20% safety margin and test fit/no-fit cases.
4. Audit a toy graph's operators against two backends.
5. Quantize/dequantize sensor features to INT8 and measure error.
6. Compare parameter reduction from distillation/pruning/quantization.
7. Calculate energy/inference from power and latency.
8. Calculate streaming sensor window sizes and duty cycles.
9. Design a ring-buffer memory layout.
10. Create a deployment checklist for CPU vs accelerator fallback.

Challenge:
- export a tiny model to ONNX and run it with ONNX Runtime CPU
- compare FP32 vs quantized model bytes/accuracy/latency using a tiny safe dataset
