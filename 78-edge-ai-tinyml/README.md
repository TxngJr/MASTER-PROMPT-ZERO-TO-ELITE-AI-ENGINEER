# Chapter 78 — Edge AI & TinyML

## 1. Edge AI Changes the Optimization Target

Cloud training asks: can we train the model?

Edge deployment asks:
- does it fit?
- are operators supported?
- is latency low enough?
- is power acceptable?
- does it work offline/reliably?

## 2. Learning Objectives

- calculate parameter/tensor storage
- estimate peak runtime memory
- distinguish Flash/storage from RAM/activation memory
- understand static memory arenas
- audit operator support
- understand execution providers/accelerators
- implement simple INT8 affine quantization
- reason about quantization/pruning/distillation together
- measure latency, power and energy per inference
- understand sensor-window streaming
- understand TinyML duty cycling
- reason about ONNX Runtime Mobile/IoT deployment
- understand LiteRT/TinyML-style deployment concepts

## 3. Edge vs TinyML

Edge AI can include:
- phones
- Raspberry Pi-class systems
- embedded Linux
- Jetson/NPUs
- browsers

TinyML usually targets tighter MCU-class budgets with small RAM/Flash and limited operator/runtime support.

## 4. Memory Types

Separate:

### Model storage
weights/graph stored in Flash/storage.

### Runtime RAM
- activations
- tensor arena
- input/output buffers
- runtime metadata

A model file fitting in Flash does not prove it fits in RAM.

## 5. Tensor Storage

For dense tensor:

~~~text
bytes = number_of_elements * bytes_per_element
~~~

FP32 uses 4 bytes/value; INT8 uses 1 byte/value.

Packed INT4 ideally approaches 0.5 byte/value before metadata/alignment overhead.

## 6. Peak Runtime Memory

Educational estimate:

~~~text
peak
=
model/runtime-resident bytes
+ peak activations
+ arena/buffers
+ runtime overhead
~~~

Actual runtimes can reuse activation buffers and have allocator/alignment overhead.

## 7. Safety Margin

Do not target 100% of device RAM.

~~~text
safe_budget = available_memory * safety_fraction
~~~

Leave room for stack, OS/runtime, drivers, application code and fragmentation.

## 8. Operator Support

An edge accelerator/runtime only speeds operators it supports.

If unsupported operations fall back to another backend, partition/copy overhead may erase gains.

Audit:
- operator types
- shapes
- dtypes
- dynamic dimensions
- quantized kernels

## 9. ONNX Runtime Mobile

Current ONNX Runtime mobile docs support CPU by default and execution providers such as XNNPACK, NNAPI and CoreML depending on platform. They explicitly recommend measuring binary size, model size, latency and power. (see references.md)

Execution-provider support is model/device-specific.

## 10. ONNX Runtime IoT / Edge

ONNX Runtime's current IoT/edge guidance emphasizes local privacy/offline/latency benefits while also warning about model-size and hardware-processing constraints. (see references.md)

## 11. Minimal Runtime Builds

Current ONNX Runtime docs note that reduced-operator prebuilt mobile packages are no longer provided from version 1.19 onward; use full operator packages or build a custom runtime for the required operator set. (see references.md)

This makes operator inventories part of deployment engineering.

## 12. Quantization

INT8 affine quantization:

~~~text
real ≈ scale * (quantized - zero_point)
~~~

Current ONNX Runtime quantization documentation supports 8-bit linear quantization and also documents INT4/UINT4 paths for supported operations/models. (see references.md)

## 13. Calibration

Post-training static quantization needs representative calibration data for activation ranges.

Calibration data should cover realistic operating conditions.

## 14. Quantization-Aware Training

QAT simulates quantization effects during training to recover quality that may be lost in post-training quantization.

## 15. Pruning + Distillation

Chapter 70 techniques become especially useful at the edge:

~~~text
large teacher
↓ distill
small student
↓ structured prune
smaller graph
↓ quantize
edge artifact
~~~

Benchmark the final artifact on target-like hardware.

## 16. Architecture Matters

Mobile/TinyML-friendly designs favor:
- depthwise/separable convolutions
- small channel counts
- simple supported activations
- static shapes where practical

But architecture choice depends on runtime/accelerator operator support.

## 17. Execution Providers / Accelerators

Possible targets:
- CPU
- XNNPACK
- NNAPI
- CoreML
- OpenVINO
- vendor NPUs/DSPs

Do not assume an accelerator path is active; profile provider assignment/fallback.

## 18. Latency Benchmarking

Measure:
- cold start
- warm latency
- p50/p95/p99 where applicable
- preprocessing
- model inference
- postprocessing

Synchronize asynchronous accelerators before recording timing.

## 19. Energy per Inference

~~~text
energy_joules = average_power_watts * latency_seconds
~~~

Then:

~~~text
inferences/joule = 1 / energy_joules
~~~

Power measurement methodology must be documented.

## 20. Duty Cycling

Always-on sensor systems can sleep between inference windows.

~~~text
duty_cycle = active_time / period
~~~

Lower duty cycle can reduce average power, subject to wake-up/sensor costs.

## 21. Sensor Windows

For sample rate f and window milliseconds W:

~~~text
samples = f * W / 1000
~~~

Streaming models may use overlapping windows and ring buffers.

## 22. Static Tensor Arena

Microcontroller runtimes often avoid general dynamic allocation and reserve a tensor arena/scratch region.

Plan peak intermediate tensors carefully.

## 23. Streaming / Ring Buffers

Instead of allocating a new full window each step:
- keep a fixed-size circular buffer
- append new sensor samples
- reuse memory

This lowers allocation pressure.

## 24. Offline / Privacy Benefits

On-device inference can:
- continue offline
- reduce network latency
- keep raw inputs local

But device compromise, local storage and update security still matter.

## 25. Model Updates

Edge fleet deployment needs:
- signed/versioned artifacts
- compatibility checks
- staged rollout
- rollback
- telemetry within privacy policy

## 26. Edge LLMs

Small language/multimodal models add:
- quantized weight memory
- KV cache
- tokenizer assets
- prompt/context budget
- thermal throttling

Do not estimate only model-file size.

## 27. TinyML Evaluation

Track:
- accuracy/task metric
- Flash/model bytes
- RAM peak
- latency
- power/energy
- operator coverage
- cold start

One aggregate score hides deployment blockers.

## 28. From Scratch

`src/edge_ai.py` implements:
- tensor_storage_bytes
- model_parameter_bytes
- peak_memory_bytes
- fits_memory_budget
- operator_coverage
- affine_int8_quantize
- affine_int8_dequantize
- energy_per_inference_mj
- inferences_per_joule
- sensor_window_samples
- duty_cycle

## 29. Common Mistakes

1. model file size confused with peak RAM
2. no RAM safety margin
3. unsupported operator fallback ignored
4. accelerator assumed active without profiling
5. quantized model not re-evaluated
6. cold start omitted
7. preprocessing excluded from latency
8. power compared under different workloads
9. dynamic allocation used unnecessarily on tiny targets
10. cloud model copied directly to MCU without redesign

## 30. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 31. Checklist

- [ ] model bytes
- [ ] RAM/activation budget
- [ ] operator coverage
- [ ] quantization
- [ ] execution provider
- [ ] latency
- [ ] energy
- [ ] sensor windows
- [ ] static arena
- [ ] deployment lifecycle

## 32. What's Next

Batch 27 closes the course with paper implementation and a full end-to-end LLM capstone/final audit.