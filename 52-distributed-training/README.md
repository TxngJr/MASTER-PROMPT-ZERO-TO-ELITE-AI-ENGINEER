# Chapter 52 — Distributed Training

## 1. Why Distributed Training?

A single accelerator eventually runs out of:
- compute throughput
- memory
- feasible training time

Distributed training splits work across multiple processes/devices.

~~~text
training job
├─ rank 0 → GPU 0
├─ rank 1 → GPU 1
├─ rank 2 → GPU 2
└─ rank 3 → GPU 3
~~~

The central problem is not merely "use more GPUs"; it is deciding **what is replicated, what is sharded, and what must communicate**.

## 2. Learning Objectives

By the end of this chapter you should be able to:

- define rank, local rank, world size and process group
- explain collective communication
- derive data-parallel gradient averaging
- distinguish all-reduce, all-gather, reduce-scatter and broadcast
- explain DDP
- shard samples across ranks correctly
- reason about global/effective batch size
- explain communication/computation overlap
- distinguish data, tensor, pipeline and model parallelism
- explain FSDP / ZeRO stages
- estimate sharded model-state memory
- explain pipeline bubbles
- reason about interconnect/topology costs
- distinguish strong and weak scaling
- understand distributed checkpointing concerns

## 3. Core Vocabulary

### Rank

Unique process ID:

~~~text
0 <= rank < world_size
~~~

### World Size

Total number of participating processes.

### Local Rank

Process/device index within one machine.

### Process Group

Set of ranks participating in collectives.

## 4. Data Parallelism

Every rank owns a full model replica.

~~~text
rank 0:
model copy + batch shard A

rank 1:
model copy + batch shard B
~~~

Each rank:
1. forward
2. backward
3. synchronize gradients
4. optimizer step

If replicas begin equal and use equal synchronized gradients, parameters remain equal.

## 5. DDP

PyTorch DistributedDataParallel synchronizes gradients across model replicas.

Important: DDP does **not** automatically split the input dataset for you.

The training code must ensure ranks receive appropriate sample shards.

Typical single-node GPU layout:
- one process per GPU
- one DDP replica per process

## 6. Gradient Averaging

Suppose rank r computes local gradient:

~~~text
g_r
~~~

For P ranks:

~~~text
g_global =
(1/P)
Σ_(r=0)^(P-1) g_r
~~~

Then every replica applies the same optimizer update.

## 7. All-Reduce

Conceptually:

~~~text
rank 0: g0 ─┐
rank 1: g1 ─┼→ reduce(sum) → distribute result
rank 2: g2 ─┼→
rank 3: g3 ─┘

all ranks receive:
g0 + g1 + g2 + g3
~~~

Frameworks commonly average or scale appropriately around this collective.

## 8. Broadcast

One rank sends the same tensor/value to every rank.

~~~text
rank 0 value
↓
all ranks
~~~

Useful for:
- initialization
- metadata
- selected state

## 9. All-Gather

Each rank begins with one shard:

~~~text
rank 0: A
rank 1: B
rank 2: C
~~~

After all-gather every rank has:

~~~text
[A, B, C]
~~~

FSDP often needs parameter all-gathers before computation.

## 10. Reduce-Scatter

Conceptually:
1. reduce corresponding contributions
2. scatter result shards

~~~text
full reduced tensor
↓
rank 0 gets shard 0
rank 1 gets shard 1
...
~~~

Useful when the final result should remain sharded.

## 11. Ring All-Reduce

A common bandwidth-efficient all-reduce can be decomposed into:
- reduce-scatter
- all-gather

For a tensor of N bytes across P ranks, idealized per-rank communicated payload is approximately:

~~~text
2 * (P-1)/P * N
~~~

Real performance also depends on:
- latency
- topology
- protocol
- contention

## 12. Global Batch Size

If each rank processes B examples and there are P ranks:

~~~text
global_batch =
B * P
~~~

With gradient accumulation A:

~~~text
effective_global_batch =
B * P * A
~~~

For language models:

~~~text
tokens_per_update =
B * P * A * sequence_length
~~~

## 13. Dataset Sharding

A simple deterministic shard:

~~~text
indices[rank::world_size]
~~~

Real samplers also handle:
- shuffle
- epochs
- drop/pad behavior
- equal numbers of steps

Unequal rank lengths can cause hangs if one rank leaves a collective early.

## 14. Synchronization Cost

Data-parallel compute per rank decreases with more devices, but gradient communication remains.

If communication dominates:

~~~text
more GPUs
≠
faster training
~~~

## 15. Communication / Backward Overlap

DDP can bucket gradients.

As backward computes earlier parameter gradients, completed buckets can begin reduction while later backward kernels continue.

~~~text
backward compute
████████████████

gradient communication
     ███████████
~~~

Overlap hides some communication cost.

## 16. Strong Scaling

Fixed total problem size.

Increase devices and ask:

> How much faster does the same job finish?

Ideal:

~~~text
speedup(P) = P
~~~

Real efficiency:

~~~text
efficiency =
speedup(P) / P
~~~

## 17. Weak Scaling

Increase:
- devices
- problem size

while keeping work/device roughly constant.

Measures ability to grow workload with hardware.

## 18. Model Parallelism

When the model itself does not fit one device, split model state/computation.

Major families:
- tensor parallel
- pipeline parallel
- fully sharded data parallel
- expert parallel for MoE

## 19. Tensor Parallelism

Split large tensor operations across devices.

Example column split:

~~~text
W =
[W0 | W1]

rank 0 computes xW0
rank 1 computes xW1
↓
gather/compose outputs
~~~

Attention/MLP matrices can be partitioned along suitable dimensions.

## 20. Communication in Tensor Parallelism

Tensor parallelism may require collectives inside each model layer.

This makes fast interconnect especially important.

Data parallelism often communicates around backward gradient synchronization; tensor parallelism can communicate repeatedly during forward/backward.

## 21. Pipeline Parallelism

Split layers into stages:

~~~text
GPU 0: layers 0–7
      ↓ activations
GPU 1: layers 8–15
      ↓
GPU 2: layers 16–23
~~~

Use micro-batches to keep stages busy.

## 22. Pipeline Bubble

For P stages and M micro-batches, a simplified forward-only utilization model is:

~~~text
efficiency ≈
M / (M + P - 1)
~~~

More micro-batches reduce bubble fraction, but increase scheduling/activation complexity.

## 23. FSDP / ZeRO Motivation

Classic data parallelism replicates:
- parameters
- gradients
- optimizer states

These may exceed device memory.

Sharding model states across ranks reduces per-rank memory.

## 24. ZeRO-Style Stages

Conceptually:

### Stage 1
Shard optimizer states.

### Stage 2
Shard optimizer states + gradients.

### Stage 3
Shard optimizer states + gradients + parameters.

This is a conceptual memory decomposition; concrete framework implementations differ.

## 25. FSDP FULL_SHARD

Current PyTorch FSDP FULL_SHARD conceptually:
- shards parameters
- shards gradients
- shards optimizer states
- all-gathers parameters when needed
- reduce-scatters gradients after backward

Memory saving is exchanged for communication.

## 26. State Memory Estimate

Let:
- P_b = parameter bytes
- G_b = gradient bytes
- O_b = optimizer-state bytes
- W = world size

Replicated data parallel:

~~~text
per_rank ≈ P_b + G_b + O_b
~~~

Fully sharded idealized:

~~~text
per_rank ≈
(P_b + G_b + O_b) / W
~~~

Peak runtime memory can be higher due to temporary all-gathers, activations and communication buffers.

## 27. Activation Memory

FSDP/ZeRO primarily target model state.

Activations still consume memory.

Combine with:
- activation checkpointing
- sequence parallelism
- smaller micro-batches
- mixed precision

## 28. Checkpointing

Distributed checkpoints must define:
- whether state is sharded
- rank ownership
- optimizer shards
- metadata
- resharding when world size changes

Do not assume a local rank-0 checkpoint contains every required shard.

## 29. Backends

Common PyTorch distributed backends include:
- NCCL for NVIDIA GPU collectives
- Gloo for CPU and some distributed scenarios

For NVIDIA GPU distributed training, current PyTorch documentation recommends NCCL.

## 30. Multi-Node Topology

Communication paths may include:
- intra-GPU high-bandwidth links
- PCIe
- CPU/socket boundaries
- network adapters
- Ethernet / InfiniBand-class fabrics

Parallelism strategy should account for topology.

A common principle:
- communication-heavy tensor parallel inside a fast node/domain
- data parallel across slower links

but measure your actual hardware.

## 31. Failure Modes

Distributed bugs can become:
- deadlocks
- collective mismatch
- rank desynchronization
- duplicated samples
- missing samples
- wrong batch scaling
- checkpoint corruption

Always debug with tiny world sizes first.

## 32. From Scratch

src/distributed_numpy.py includes:

- shard_indices
- all_reduce_mean
- ring_allreduce_per_rank_bytes
- effective_global_batch
- zero_style_state_bytes
- pipeline_efficiency
- scaling_efficiency

## 33. Common Mistakes

1. every rank receives identical samples unintentionally
2. local batch confused with global batch
3. loss scaling inconsistent with gradient averaging
4. one rank skips a collective
5. random seeds create unwanted identical augmentation
6. using tensor parallel across a slow network blindly
7. memory estimate ignores activations/temp buffers
8. only rank 0 saves incomplete sharded state
9. benchmarking without warmup/synchronization
10. assuming 2 GPUs means exactly 2× speed

## 34. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 35. Checklist

- [ ] rank / world size
- [ ] collectives
- [ ] DDP
- [ ] data sharding
- [ ] global batch
- [ ] all-reduce
- [ ] tensor parallel
- [ ] pipeline parallel
- [ ] FSDP / ZeRO
- [ ] communication cost
- [ ] scaling efficiency
- [ ] distributed checkpoints

## 36. What's Next

Chapter 53 goes below the framework layer into the CUDA execution and GPU memory model.
