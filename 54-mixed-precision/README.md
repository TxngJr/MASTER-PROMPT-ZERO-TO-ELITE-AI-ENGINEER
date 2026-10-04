# Chapter 54 — Mixed Precision: FP32, FP16 & BF16

## 1. Why Mixed Precision?

Training in full FP32 is robust but uses more memory/bandwidth than lower-precision arithmetic.

Mixed precision keeps numerically sensitive work in safer precision while running suitable operations in lower precision.

~~~text
FP32 model/training state
       ↓
autocast chooses operation dtypes
       ↓
FP16/BF16 compute where safe
FP32 compute where needed
       ↓
stable backward/update
~~~

## 2. Learning Objectives

- understand sign/exponent/fraction fields
- compare FP32, FP16 and BF16
- explain dynamic range vs precision
- identify overflow and underflow
- explain why FP16 often needs loss scaling
- understand BF16's larger exponent range
- explain autocast
- use current torch.amp APIs
- explain GradScaler behavior
- unscale before gradient clipping
- understand FP32 optimizer/master-state concepts
- estimate memory savings
- debug NaN/Inf mixed-precision failures

## 3. Floating-Point Structure

Conceptually:

~~~text
value ≈ (-1)^sign × significand × 2^exponent
~~~

More exponent bits increase representable range. More fraction/mantissa bits increase local precision.

## 4. FP32

Typical IEEE binary32:

~~~text
1 sign
8 exponent
23 fraction bits
~~~

Strengths:
- wide range
- relatively high precision
- robust default training dtype

Cost:
- 4 bytes/element

## 5. FP16

IEEE binary16:

~~~text
1 sign
5 exponent
10 fraction bits
~~~

Advantages:
- 2 bytes/element
- high accelerator throughput on suitable hardware

Challenge:
- much smaller exponent range than FP32/BF16
- gradients can underflow to zero
- large values can overflow to Inf

## 6. BF16

BFloat16:

~~~text
1 sign
8 exponent
7 fraction bits
~~~

BF16 keeps the FP32-sized exponent field, giving it a similar broad dynamic range but less significand precision.

This often makes BF16 easier to train with than FP16 on supported hardware, although rounding error is larger for many values.

## 7. Range vs Precision

Key distinction:

~~~text
FP16:
more fraction precision than BF16
but much smaller numeric range

BF16:
less fraction precision
but FP32-like exponent range
~~~

## 8. Underflow

Very small non-zero values may round to zero.

In FP16 this can affect tiny gradients:

~~~text
true gradient
1e-8
↓ cast FP16
0
~~~

Once rounded to zero, the optimizer cannot recover that gradient.

## 9. Overflow

Values above the finite representable maximum become Inf.

Then operations can produce:
- Inf
- NaN
- invalid optimizer updates

## 10. Loss Scaling

FP16 gradient scaling multiplies the loss before backward:

~~~text
scaled_loss = loss × S
~~~

Then:

~~~text
scaled_grad = true_grad × S
~~~

This moves tiny gradients into FP16's representable range.

Before optimizer update:

~~~text
true_grad = scaled_grad / S
~~~

## 11. Static vs Dynamic Scaling

Static:
- fixed scale chosen manually

Dynamic:
- increase scale after stable steps
- reduce scale after Inf/NaN
- skip unsafe optimizer updates when overflow is detected

## 12. Gradient Clipping Order

If using scaled gradients:

~~~text
backward on scaled loss
↓
unscale gradients
↓
clip true gradient norm
↓
optimizer step
~~~

Clipping before unscale uses the wrong scale.

## 13. Autocast

Current PyTorch AMP uses torch.amp.autocast with a device type.

~~~python
with torch.amp.autocast('cuda', dtype=torch.float16):
    output = model(inputs)
    loss = loss_fn(output, targets)
~~~

Autocast selects operation-specific dtypes. Do not manually call half() on the whole model just because autocast is enabled.

## 14. Backward Placement

Current PyTorch guidance wraps forward + loss in autocast and performs backward after leaving the autocast context.

Backward operations then follow the dtypes chosen by the corresponding forward operations.

## 15. Current GradScaler API

For CUDA FP16 training, current PyTorch uses:

~~~python
scaler = torch.amp.GradScaler('cuda')
~~~

Older torch.cuda.amp GradScaler/autocast entry points are deprecated in current documentation.

## 16. Typical CUDA FP16 Loop

~~~python
optimizer.zero_grad(set_to_none=True)

with torch.amp.autocast('cuda', dtype=torch.float16):
    logits = model(x)
    loss = loss_fn(logits, y)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
~~~

## 17. BF16 Autocast

On supported hardware:

~~~python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    ...
~~~

BF16's exponent range often reduces the need for gradient scaling compared with FP16. Whether scaling is useful still depends on hardware/workload.

## 18. CPU BF16

PyTorch also supports CPU autocast on supported operations:

~~~python
with torch.amp.autocast('cpu', dtype=torch.bfloat16):
    output = model(x)
~~~

This is useful for CI smoke tests even when CUDA is unavailable.

## 19. FP32 Master / Optimizer State

Mixed precision does not mean every training tensor must be low precision.

Training systems may retain higher precision for:
- optimizer states
- parameter updates/master values
- reductions
- sensitive normalization/loss operations

Exact behavior depends on framework/parallelism strategy.

## 20. Memory

Raw tensor storage:

~~~text
FP32 = 4 bytes/value
FP16 = 2 bytes/value
BF16 = 2 bytes/value
~~~

But total training memory also includes:
- gradients
- optimizer states
- activations
- temporary kernels
- communication buffers

Therefore mixed precision does not automatically halve total training memory.

## 21. Tensor Cores

Modern NVIDIA GPUs can accelerate matrix operations in lower precision using specialized tensor-core paths when dimensions/layouts/operations are suitable.

Actual speedup depends on workload and hardware.

## 22. Precision-Sensitive Operations

Some operations benefit from FP32 accumulation/computation, for example reductions or certain normalization/loss operations. Autocast maintains an operation-specific eligibility policy rather than blindly forcing one dtype everywhere.

## 23. Detecting Overflow

Useful checks:
- loss finite
- gradient norm finite
- parameter values finite
- scaler trend

Repeated scale backoff can indicate numerical instability.

## 24. Reproducibility

Changing dtype/kernel can change rounding and reduction order. Numerically valid mixed-precision runs are not guaranteed bitwise identical to FP32.

## 25. From Scratch

src/mixed_precision.py includes:

- floating_format_info
- bfloat16_roundtrip
- scale_gradients
- unscale_gradients
- contains_nonfinite
- update_dynamic_loss_scale
- tensor_storage_bytes
- relative_error

## 26. Common Mistakes

1. thinking FP16 and BF16 have the same range
2. converting the entire model manually while also using autocast
3. backward inside autocast unnecessarily
4. gradient clipping before unscale
5. optimizer step after Inf gradients
6. assuming BF16 has FP32 precision
7. assuming 16-bit tensors halve total training memory
8. using deprecated AMP namespaces in new code
9. comparing speed without synchronization/warmup
10. ignoring scaler behavior and non-finite diagnostics

## 27. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Checklist

- [ ] FP32 / FP16 / BF16 formats
- [ ] range vs precision
- [ ] underflow / overflow
- [ ] autocast
- [ ] loss scaling
- [ ] GradScaler
- [ ] unscale → clip → step
- [ ] BF16 behavior
- [ ] optimizer/master state
- [ ] memory accounting
- [ ] stability debugging

## 29. What's Next

Batch 19 moves from pretraining systems into model adaptation: full fine-tuning, LoRA/QLoRA/PEFT and instruction tuning.