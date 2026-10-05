# Precision Formats: FP64, FP32, TF32, FP16, BF16, FP8 and INT8

Precision affects range, resolution, memory traffic, arithmetic throughput and numerical stability.

| Format | Typical role |
|---|---|
| FP64 | scientific/high-precision reference work; expensive on many AI accelerators |
| FP32 | standard single precision; common baseline |
| TF32 | NVIDIA tensor-core math mode with FP32 range and reduced mantissa precision for selected matrix operations |
| FP16 | compact floating point with limited exponent range; often needs loss scaling |
| BF16 | FP32-like exponent range with fewer mantissa bits; often easier for deep-learning training |
| FP8 | very low precision floating point used on supported modern accelerators with scaling strategies |
| INT8 | integer quantization, primarily common for inference and some specialized training paths |

## Range vs precision
Exponent bits largely control dynamic range; mantissa/significand bits control local precision. Therefore BF16 can represent a much wider magnitude range than FP16 even though both occupy 16 bits.

## Mixed-precision rule
Keep numerically sensitive operations or master state in a safer precision when needed, use lower precision for high-throughput tensor operations, and verify gradients/losses remain finite.

## TF32 note
TF32 is not a storage dtype users typically allocate like FP32; it is a compute mode for supported Tensor Core operations.

## FP8 note
FP8 requires hardware/framework support and scale management. This laptop curriculum should treat FP8 conceptually unless the installed hardware reports support.

## INT8 note
INT8 is covered deeply in Chapter 61 quantization. Do not confuse quantized inference with FP16/BF16 autocast training.
