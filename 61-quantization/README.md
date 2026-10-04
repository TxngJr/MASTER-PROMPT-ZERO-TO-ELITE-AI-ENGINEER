# Chapter 61 — Quantization: INT8, INT4, GPTQ, AWQ & GGUF

## 1. Why Quantization?

Large language models are often limited by:
- parameter memory
- memory bandwidth
- cache capacity
- device placement

Quantization stores values using fewer bits while trying to preserve model quality.

~~~text
FP32 / FP16 weights
↓
quantizer
↓
INT8 / INT4 / other compact representation
↓
dequantize or low-bit kernel
↓
inference
~~~

## 2. Learning Objectives

You should be able to:

- distinguish quantization from mixed precision
- derive symmetric quantization
- derive asymmetric quantization
- explain zero points
- compare per-tensor / per-channel / grouped quantization
- calculate quantization error
- understand weight-only vs weight+activation quantization
- explain calibration
- understand static vs dynamic quantization
- explain GPTQ conceptually
- explain AWQ conceptually
- distinguish quantization algorithms from GGUF file/runtime formats
- estimate storage/memory
- reason about quality-speed-memory trade-offs
- understand current Transformers quantization options

## 3. Quantization vs Mixed Precision

Mixed precision:

~~~text
compute/storage dtype
=
FP32 / FP16 / BF16 combinations
~~~

Quantization:

~~~text
continuous floating-point values
↓
finite low-bit codebook / integer levels
~~~

A 4-bit quantized weight may still be dequantized into FP16/BF16 for computation.

## 4. Uniform Quantization

Given a scale s and integer code q:

~~~text
x_hat =
s * q
~~~

Quantization chooses q near:

~~~text
q ≈ round(x / s)
~~~

Then clamps to representable integer range.

## 5. Symmetric Quantization

For signed b-bit integers:

~~~text
q_max =
2^(b-1)-1
~~~

One common symmetric scale:

~~~text
s =
max(|x|)
/
q_max
~~~

Then:

~~~text
q =
clip(
round(x/s),
-q_max,
q_max
)
~~~

Dequantize:

~~~text
x_hat =
s q
~~~

Zero maps naturally to integer zero.

## 6. Asymmetric Quantization

Represent:

~~~text
x_hat =
s (q - z)
~~~

where:
- s = scale
- z = zero point

This allows the integer code range to shift toward an asymmetric floating range.

Useful when min/max are not symmetric around zero.

## 7. Zero Point

Zero point aims to map real zero exactly:

~~~text
0
≈
s(q_zero - z)
~~~

Therefore:

~~~text
q_zero ≈ z
~~~

Zero-point handling can add arithmetic/kernel complexity.

## 8. Per-Tensor Quantization

One scale for an entire tensor.

Pros:
- simple
- tiny metadata

Cons:
- one outlier can enlarge scale
- small-value regions may lose resolution

## 9. Per-Channel Quantization

Different scale per output/input channel.

Pros:
- adapts to channel ranges
- often better quality

Cons:
- extra metadata
- more complex kernels

For linear weights, output-channel quantization is common.

## 10. Grouped Quantization

Split weight vectors into groups:

~~~text
group 0 → scale_0
group 1 → scale_1
...
~~~

Group size trades:
- metadata
- quality
- kernel efficiency

Modern 4-bit LLM quantization often uses grouped/blockwise approaches.

## 11. Quantization Error

For original x and reconstructed x_hat:

~~~text
error =
x - x_hat
~~~

Metrics:
- MSE
- max absolute error
- relative error
- downstream task loss/accuracy

Tensor reconstruction error alone does not fully predict LLM quality.

## 12. Clipping

Outliers may dominate scale.

Quantizers may deliberately clip extremes to improve resolution for most values.

This trades:
- outlier distortion
for
- lower average error

Calibration chooses such trade-offs.

## 13. Calibration

Post-training quantization may run representative data through the model to estimate:
- activation ranges
- important channels
- Hessian/statistical information
- clipping thresholds

Calibration data should match deployment distribution where possible.

## 14. Weight-Only Quantization

Store weights in low precision while activations remain FP16/BF16/FP32-like.

Advantages:
- large model-memory saving
- simpler than quantizing activations

Especially useful for autoregressive decode where weight bandwidth is expensive.

## 15. Activation Quantization

Quantize activations too.

Potential benefits:
- reduced bandwidth
- low-bit matrix kernels

Challenges:
- activation outliers
- dynamic ranges
- sensitive layers

INT8 weight+activation is common in suitable deployments.

## 16. Static vs Dynamic

Static:
- quantization parameters prepared ahead of inference

Dynamic:
- some quantization parameters computed at runtime

Exact terminology varies by framework.

Always inspect the specific backend.

## 17. Post-Training Quantization

PTQ quantizes a pretrained model without full retraining.

Examples/ideas:
- simple min/max
- GPTQ
- AWQ

Goal:
- retain quality
- avoid expensive retraining

## 18. Quantization-Aware Training

QAT simulates quantization effects during training so the model can adapt.

More expensive than PTQ but can improve low-bit accuracy.

## 19. GPTQ Concept

GPTQ is a post-training weight quantization method that uses approximate second-order information to reduce reconstruction error layer-by-layer.

Conceptually:

~~~text
layer weights
+
calibration activations
↓
quantize weights
↓
compensate remaining weights
↓
next quantization decisions
~~~

Do not reduce GPTQ to naive round(weight/scale).

## 20. AWQ Concept

Activation-aware Weight Quantization observes activation behavior to identify especially important weights/channels.

The method protects/scales important weights before aggressive low-bit quantization.

Current Transformers documentation supports loading AWQ models and describes AWQ as 4-bit activation-aware weight quantization. citeturn912558search1

## 21. bitsandbytes

Current Transformers supports:
- 8-bit
- 4-bit

through bitsandbytes integrations.

This is a runtime/library path, not a universal quantization algorithm.

## 22. GGUF

GGUF is a model file format/ecosystem used prominently by llama.cpp.

It can contain:
- model metadata
- tensors
- tokenizer metadata
- tensors stored in multiple quantized types

Important:

> GGUF is not itself one quantization objective like GPTQ or AWQ.

A GGUF file may contain different quantized tensor formats.

## 23. llama.cpp Quantization

llama.cpp focuses on efficient local CPU/GPU inference and supports many low-bit representations in GGUF workflows.

Current llama.cpp can run quantized GGUF models across CPU and multiple accelerator backends. citeturn912558search2

## 24. Storage Math

For N raw parameters at b bits:

~~~text
ideal_weight_bytes =
N * b / 8
~~~

Examples:

~~~text
1B parameters

FP16:
~2 GB raw weights

INT8:
~1 GB

INT4:
~0.5 GB
~~~

Real files use extra:
- scales
- zero points
- grouping metadata
- tensor metadata
- alignment

## 25. KV Cache Is Separate

Quantizing model weights does not automatically quantize KV cache.

Inference memory often includes:

~~~text
weights
+
KV cache
+
activations
+
runtime buffers
~~~

Long contexts can make KV cache dominate.

## 26. Dequantization

Low-bit weights may be:
- unpacked/dequantized before matrix multiply
- dequantized inside fused kernels
- consumed directly by specialized integer/low-bit kernels

Runtime implementation determines performance.

## 27. Quantization Can Be Slower

Smaller weights do not guarantee faster inference.

Possible bottlenecks:
- dequantization overhead
- poor kernel support
- small batch
- unsupported hardware
- conversion overhead

Benchmark the actual model/runtime/device.

## 28. Layer Sensitivity

Some layers tolerate quantization poorly.

Strategies:
- keep selected modules high precision
- use per-channel/group scales
- mixed bit widths
- calibration-aware methods

Current Transformers quantization configs expose modules that should not be converted for some methods. citeturn912558search0

## 29. From Scratch

src/quantization_numpy.py implements:

- signed_integer_range
- symmetric_quantize
- symmetric_dequantize
- asymmetric_quantize
- asymmetric_dequantize
- per_channel_symmetric_quantize
- grouped_symmetric_quantize
- quantization_mse
- ideal_storage_bytes
- compression_ratio

## 30. Common Mistakes

1. calling FP16 "INT16 quantization"
2. saying GGUF equals one quantization algorithm
3. ignoring scales/metadata in memory estimates
4. assuming INT4 is always 4× faster than FP16
5. evaluating only tensor MSE
6. calibrating on unrelated data
7. per-tensor scale dominated by outliers
8. quantizing sensitive modules blindly
9. forgetting KV-cache memory
10. comparing runtimes with different context/batch settings

## 31. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 32. Checklist

- [ ] symmetric/asymmetric
- [ ] scale / zero point
- [ ] per-tensor/channel/group
- [ ] calibration
- [ ] weight-only
- [ ] activation quantization
- [ ] GPTQ
- [ ] AWQ
- [ ] GGUF
- [ ] memory math
- [ ] quality evaluation

## 33. What's Next

Chapter 62 optimizes autoregressive inference itself: KV caches, batching, paged memory, FlashAttention and speculative decoding.
