# Chapter 70 — Model Compression: Pruning, Distillation & Quantization

## 1. Why Compress?

A model can be accurate yet too expensive for deployment.

Compression targets one or more of:
- parameter storage
- memory bandwidth
- activation memory
- FLOPs
- latency
- energy
- hardware footprint

Compression quality is a Pareto trade-off, not a file-size contest.

## 2. Learning Objectives

- measure sparsity correctly
- implement global magnitude pruning
- distinguish unstructured and structured pruning
- understand masks vs physically smaller tensors
- explain why sparse weights may not produce speedup
- derive knowledge-distillation temperature scaling
- compute teacher/student soft targets
- combine supervised and distillation objectives
- distinguish response, feature and relation distillation
- reason about pruning + quantization + distillation combinations
- design a compression evaluation protocol

## 3. Unstructured Pruning

Remove individual weights:

~~~text
W_ij -> 0
~~~

A magnitude rule prunes weights with smallest absolute magnitude.

Result:
- many zeros
- unchanged dense tensor shape

## 4. Sparsity

~~~text
sparsity = number_of_zeros / number_of_elements
density  = 1 - sparsity
~~~

Always distinguish target sparsity, achieved sparsity, storage format and runtime sparse-kernel support.

## 5. Global vs Layerwise Pruning

Layerwise pruning gives every layer its own pruning fraction.

Global pruning ranks weights across selected layers and prunes globally smallest values.

## 6. Structured Pruning

Remove entire structures such as neurons, channels, attention heads, rows, columns or blocks.

Structured pruning can change dense matrix dimensions and is often easier for ordinary hardware to accelerate.

## 7. Masking Is Not Compression Yet

A dense FP16 tensor with 90% zeros may still occupy roughly the same dense storage.

Actual memory/runtime gains may require sparse representation, structural removal, specialized kernels or model surgery.

## 8. PyTorch Pruning

Current PyTorch documentation still provides `torch.nn.utils.prune` utilities and an official pruning tutorial for sparsifying modules. citeturn744269search4

PyTorch pruning commonly uses masks/reparameterization until pruning is made permanent.

## 9. Iterative Pruning

~~~text
train
-> prune a little
-> fine-tune
-> prune again
-> fine-tune
~~~

Iterative pruning often preserves quality better at high sparsity than a single aggressive step.

## 10. Knowledge Distillation

Teacher:
~~~text
large / accurate
~~~

Student:
~~~text
smaller / cheaper
~~~

Current PyTorch's official distillation tutorial demonstrates adding a soft-target teacher loss to ordinary supervised training. citeturn315138search0

## 11. Temperature

Given logits z and temperature T:

~~~text
p_i(T) = exp(z_i/T) / sum_j exp(z_j/T)
~~~

Larger T produces a softer distribution and exposes relationships among non-argmax classes.

## 12. Distillation KL

A common response-distillation term:

~~~text
L_KD = T^2 * KL(p_teacher(T) || p_student(T))
~~~

The T² factor compensates for temperature-related gradient scaling in this standard formulation.

## 13. Combined Objective

~~~text
L = alpha * L_hard + (1-alpha) * L_KD
~~~

Document your alpha convention because libraries/papers may weight terms differently.

## 14. Feature Distillation

Match internal representations, optionally through a projection if teacher/student hidden sizes differ.

## 15. Relation Distillation

Match relationships such as pairwise distances, similarity matrices or attention relations instead of individual hidden vectors.

## 16. Quantization + Pruning

Quantization reduces bits/value. Pruning reduces effective parameters/connections.

They optimize different dimensions and can be combined, but order and hardware support affect final quality and speed.

## 17. Distillation + Quantization

~~~text
teacher
↓ distill
small student
↓ quantize
low-bit student
~~~

Evaluate the final low-bit artifact, not only the pre-quantized student.

## 18. Compression Metrics

Track:
- task quality
- parameter count
- nonzero count
- raw bytes
- serialized bytes
- peak memory
- latency
- throughput
- energy where possible

## 19. Compression Ratio

~~~text
compression_ratio = original_size / compressed_size
~~~

State whether size means raw weights, file size or runtime memory.

## 20. The Speedup Trap

Fewer nonzeros do not guarantee faster inference because sparse indices, unsupported patterns, kernel overhead and hardware mismatch can dominate.

## 21. From Scratch

`src/compression.py` implements:
- sparsity
- nonzero_count
- magnitude_prune
- structured_row_prune
- softmax_temperature
- kl_divergence
- distillation_loss
- compression_ratio

## 22. Common Mistakes

1. zeros in dense tensor called real storage compression
2. sparsity assumed equal to speedup
3. pruning normalization/critical layers blindly
4. no fine-tuning after aggressive pruning
5. teacher/student evaluated on different splits
6. temperature convention undocumented
7. compression ratio definition omitted
8. final quantized model not reevaluated
9. no warmup in latency benchmark
10. framework mask metadata ignored

## 23. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 24. Checklist

- [ ] unstructured pruning
- [ ] structured pruning
- [ ] sparsity / density
- [ ] iterative pruning
- [ ] teacher/student
- [ ] temperature
- [ ] KD objective
- [ ] feature distillation
- [ ] compression combinations
- [ ] final benchmark

## 25. What's Next

Chapter 71 explores sparse activation rather than sparse weights: Mixture-of-Experts routes each token through only a subset of experts.