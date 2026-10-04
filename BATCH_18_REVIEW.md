# Batch 18 Review — Chapters 52–54

## Chapters

- 52 — Distributed Training
- 53 — GPU / CUDA Fundamentals
- 54 — Mixed Precision: FP32 / FP16 / BF16

## Distributed Training Skills

- rank / local rank / world size
- process groups
- all-reduce / broadcast / all-gather / reduce-scatter
- DDP gradient synchronization
- rank-aware dataset sharding
- global/effective batch size
- gradient communication overlap
- strong / weak scaling
- tensor parallelism
- pipeline parallelism
- pipeline bubbles
- FSDP / ZeRO concepts
- model-state memory estimates
- distributed checkpoints / topology

## GPU / CUDA Skills

- host vs device
- kernel / grid / block / thread
- global thread indexing
- Streaming Multiprocessors
- 32-thread warps / SIMT
- warp divergence
- registers / local / global / shared memory
- memory coalescing
- shared-memory bank conflicts
- block synchronization
- asynchronous launches
- streams / events
- occupancy constraints
- arithmetic intensity
- roofline model
- CUDA profiling

## Mixed Precision Skills

- FP32 / FP16 / BF16 formats
- exponent range vs significand precision
- underflow / overflow
- autocast
- dynamic loss scaling
- GradScaler
- unscale → clip → step ordering
- BF16 behavior
- optimizer/master-state concepts
- memory accounting
- NaN/Inf diagnostics

## Implemented From Scratch

### Chapter 52
- shard_indices
- all_reduce_mean
- ring_allreduce_per_rank_bytes
- effective_global_batch
- zero_style_state_bytes
- pipeline_efficiency
- scaling_efficiency
- communication_time_seconds

### Chapter 53
- ceil_div
- grid_size_1d
- global_thread_index
- warp_and_lane
- transaction_segment_count
- theoretical_resident_blocks
- arithmetic_intensity
- roofline_bound

### Chapter 54
- floating_format_info
- bfloat16_roundtrip
- scale_gradients
- unscale_gradients
- contains_nonfinite
- update_dynamic_loss_scale
- tensor_storage_bytes
- relative_error

## Integration Project

Three modes:

1. distributed planning/math
2. CUDA device/runtime inspection
3. current torch.amp mixed-precision training smoke

## Current API Audit

### PyTorch Distributed

Current DDP behavior:
- synchronizes gradients across replicas
- does not automatically shard input samples
- expects process-group initialization
- commonly uses one process per GPU

Current FSDP FULL_SHARD:
- shards parameters
- shards gradients
- shards optimizer states
- uses all-gather and reduce-scatter around compute

### CUDA

Current NVIDIA programming guidance:
- warps contain 32 threads
- shared memory is block-visible scratch space
- global-memory access efficiency depends heavily on coalescing
- synchronization via block barriers is block-scoped

### AMP

Current PyTorch AMP:
- uses torch.amp.autocast(device_type=...)
- uses torch.amp.GradScaler("cuda", ...) for CUDA scaling
- old torch.cuda.amp autocast/GradScaler entry points are deprecated
- autocast wraps forward/loss; backward is performed after leaving the context

## Methodology Audit

### Distributed
- rank shards cover each sample exactly once in the educational test
- all-reduce mean matches direct gradient averaging
- DDP/global batch concepts are separated
- idealized ZeRO memory estimates explicitly exclude activation/temp-buffer peaks

### CUDA
- transaction-segment model is labeled as simplified
- occupancy estimate is resource-bound, not claimed to predict final performance
- CPU CI does not pretend to validate real NVIDIA kernel throughput

### Precision
- FP16/BF16 numeric differences are explicit
- CPU BF16 autocast gives framework coverage without requiring CUDA in CI
- CUDA path uses current torch.amp APIs when hardware exists
- gradient clipping is performed after scaler unscale
- non-finite/output checks are included

## Interpretation Audit

- more GPUs do not guarantee linear speedup
- sharding memory creates communication costs
- high occupancy does not guarantee high throughput
- coalescing and arithmetic intensity matter separately
- mixed precision does not halve total training memory automatically
- BF16's wide range does not mean FP32-level precision
- lower precision differences are expected numerical effects, not automatically bugs

## Exit Gate

Before Chapter 55:

1. Batch 18 Core CI passes
2. Batch 18 PyTorch smoke passes
3. derive data-parallel gradient averaging
4. explain all-reduce / all-gather / reduce-scatter
5. estimate DDP vs ZeRO/FSDP state memory
6. map CUDA grid/block/thread indices
7. explain warp divergence and coalescing
8. calculate roofline bounds
9. compare FP32 / FP16 / BF16 ranges
10. explain autocast and loss scaling
11. implement correct unscale → clip → step ordering
