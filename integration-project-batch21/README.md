# Batch 21 Integration Project — Quantized Inference & Serving Capacity Lab

Batch 21 connects low-bit model storage, KV-cache capacity and production serving metrics.

## Part A — Quantization

A synthetic weight matrix is quantized with:

~~~text
FP weights
├── INT8 symmetric
├── INT4 symmetric
└── INT4 per-channel
~~~

Report:
- reconstruction MSE
- ideal raw weight bytes
- compression ratio vs FP16
- per-channel scale count

## Part B — Inference Capacity

Estimate a 32-layer decoder cache under:

~~~text
MHA: 32 KV heads
vs
GQA: 8 KV heads
~~~

Also report:
- padding efficiency
- paged block fragmentation
- prefix-cache saved tokens
- speculative acceptance rate

## Part C — Serving SLO Planner

Synthetic request traces produce:
- p50 / p95 / p99 latency
- Little's-Law concurrency
- required replica estimate
- success rate
- 5% canary traffic split

## Part D — PyTorch Weight-Only Smoke

Quantize/dequantize one linear weight tensor and compare its output with the FP32 source tensor.

This is an educational reconstruction test, not a claim that NumPy quantization matches an optimized production low-bit kernel.

## Run

~~~bash
python integration-project-batch21/src/quantized_inference_serving_lab.py --mode quantization
python integration-project-batch21/src/quantized_inference_serving_lab.py --mode capacity
python integration-project-batch21/src/quantized_inference_serving_lab.py --mode serving
python integration-project-batch21/src/quantized_inference_serving_lab.py --mode torch-weight-only --bits 4
~~~

## Optional Serving Environment

~~~bash
python -m pip install -r requirements-batch21-serving.txt
~~~

vLLM and llama.cpp are hardware/runtime-specific and intentionally not forced into lightweight CPU CI.

## Required Extensions

1. actual Transformers model.generate benchmark
2. cache on/off comparison
3. GGUF metadata inspection
4. llama.cpp load test
5. vLLM benchmark on compatible GPU
6. continuous-batching simulation
7. prefix-cache hit-rate experiment
8. TTFT/TPOT per-sample traces
9. bounded-queue FastAPI service
10. canary quality + latency rollback gate
