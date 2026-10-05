# Precision Formats — FP64, FP32, TF32, FP16, BF16, FP8 and INT8

Low precision is not one thing. Separate **storage format**, **compute format**, **accumulation format**, and **quantization format**.

## Floating-point comparison

| Format | Typical role | Main characteristic |
|---|---|---|
| FP64 | scientific/high-precision reference | high precision/range, expensive |
| FP32 | standard training/reference | strong general precision |
| TF32 | NVIDIA tensor-core compute mode for FP32-style workloads | reduced mantissa precision in supported matrix operations while retaining FP32-like exponent range |
| FP16 | mixed-precision training/inference | limited exponent range; often needs loss scaling |
| BF16 | mixed-precision training | FP32-like exponent range with fewer fraction bits |
| FP8 | newer accelerator training/inference paths | very low precision; scaling/format/runtime support is critical |
| INT8 | quantized inference and some specialized compute | integer scale/zero-point representation, not floating point |

Do not infer exact speedups from the format name; real behavior depends on GPU architecture, kernels, framework and tensor shapes.

## Accumulation

A low-precision multiply may accumulate into a wider format. Document:

~~~text
input dtype
weight dtype
multiply/compute mode
accumulator dtype
master-weight dtype
optimizer-state dtype
~~~

## FP16 loss scaling

If gradients underflow in FP16, multiply the loss by scale (S), backpropagate the scaled loss, unscale gradients before clipping/update, and skip/adjust the step when non-finite values are detected.

## TF32

TF32 is a hardware compute path, not a tensor storage dtype you typically save as a checkpoint. Treat reproducibility comparisons carefully because enabling/disabling it can change numeric results.

## FP8

FP8 requires explicit scale management and supported kernels/hardware. Teach it as a system of format + scale + accumulation policy rather than "just cast to FP8."

## INT8

INT8 belongs primarily to quantization: map real values to integer codes using scale (and possibly zero point). It is conceptually different from floating-point mixed precision.

## Validation matrix

For every precision experiment record:
- hardware/runtime;
- input/weight/accumulator dtype;
- loss-scaling policy;
- reference FP32/FP64 result;
- absolute/relative error;
- overflow/underflow/NaN counts;
- throughput and memory under identical workload.
