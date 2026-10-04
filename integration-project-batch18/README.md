# Batch 18 Integration Project — Distributed, CUDA & Precision Lab

Batch 18 connects large-model systems concepts with runtime behavior.

## Part A — Distributed Planner

Pure NumPy/system math verifies:

~~~text
dataset
↓
rank shards

local gradients
↓
all-reduce mean

model-state bytes
↓
DDP / ZeRO-style stage estimates

gradient bytes
↓
ring all-reduce payload estimate
~~~

The lab reports:
- sample shard sizes
- unique sample coverage
- averaged gradient
- effective global batch
- ring payload/rank
- stage 0–3 model-state memory
- pipeline efficiency

## Part B — CUDA Inspection

PyTorch path reports:

~~~text
torch.cuda.is_available()
device count
GPU name
device memory
SM count
compute capability
~~~

When CUDA is unavailable, the lab exits the CUDA-specific part cleanly instead of failing CPU CI.

On CUDA hardware it also runs a small synchronized matrix-multiply smoke.

## Part C — Mixed Precision

One tiny network trains with:

~~~text
CUDA available:
torch.amp.autocast(cuda, FP16)
+
torch.amp.GradScaler(cuda)

CPU CI:
torch.amp.autocast(cpu, BF16)
~~~

CUDA path follows:

~~~text
forward/loss under autocast
↓
scale(loss).backward()
↓
unscale optimizer gradients
↓
clip global norm
↓
scaler.step()
↓
scaler.update()
~~~

CPU BF16 path does not require the CUDA GradScaler.

## Run

~~~bash
python integration-project-batch18/src/distributed_cuda_precision_lab.py --mode distributed
python integration-project-batch18/src/distributed_cuda_precision_lab.py --mode gpu
python integration-project-batch18/src/distributed_cuda_precision_lab.py --mode precision --steps 10
~~~

## Required Extensions

1. two-process Gloo all-reduce demo
2. multi-GPU NCCL DDP when hardware is available
3. distributed sampler epoch/shuffle audit
4. FSDP memory experiment
5. CUDA-event timing
6. profiler trace
7. FP32 vs BF16 validation comparison
8. CUDA FP16 scaler overflow experiment
9. throughput/tokens-per-second benchmark
10. integrate AMP into Batch 17 tiny LLM pretraining
