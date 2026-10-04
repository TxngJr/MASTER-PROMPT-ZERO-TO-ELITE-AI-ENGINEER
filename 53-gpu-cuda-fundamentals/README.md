# Chapter 53 — GPU & CUDA Fundamentals

## 1. Why Learn CUDA If PyTorch Already Exists?

PyTorch hides kernel launches, transfers and synchronization behind tensor APIs. When performance matters, you still need the execution and memory model.

~~~text
Python / PyTorch op
↓
CUDA runtime
↓
kernel launch
↓
grid → blocks → warps → threads
↓
SM execution + GPU memory hierarchy
~~~

## 2. Learning Objectives

- distinguish CPU and GPU execution models
- explain host vs device memory
- define kernel, grid, block and thread
- calculate global thread indices
- explain Streaming Multiprocessors
- explain 32-thread warps and SIMT
- explain warp divergence
- distinguish registers, local, shared and global memory
- reason about global-memory coalescing
- explain shared-memory bank conflicts
- understand synchronization boundaries
- explain kernel launch asynchrony
- understand streams/events
- estimate occupancy constraints
- compute arithmetic intensity and roofline bounds
- inspect CUDA devices from PyTorch

## 3. CPU vs GPU

CPU cores emphasize low-latency serial/control performance. GPUs emphasize throughput across many parallel execution lanes. A GPU does not make serial work automatically fast.

## 4. Host and Device

CPU code runs on the host; CUDA kernels execute on the device. Host-device transfers can dominate small workloads, so useful GPU systems keep data resident on the device where possible.

## 5. Kernel / Grid / Block / Thread

~~~text
kernel launch
└─ grid
   ├─ block 0
   │  ├─ thread 0
   │  └─ ...
   └─ block 1 ...
~~~

For one dimension:

~~~text
i = blockIdx.x * blockDim.x + threadIdx.x
num_blocks = ceil(N / block_size)
~~~

The final block normally guards indices outside N.

## 6. Streaming Multiprocessors

Thread blocks are scheduled onto SMs. Each SM has finite threads/warps, registers and shared-memory capacity, so resource usage per block limits how many blocks can be resident simultaneously.

## 7. Warps and SIMT

Current NVIDIA CUDA documentation defines a warp as 32 threads. Threads in a warp execute the same kernel under the SIMT model while retaining per-thread logical control flow.

## 8. Warp Divergence

When lanes in one warp choose different branches, execution may serialize subsets of lanes. Divergence is therefore most important inside a warp.

## 9. Registers and Local Memory

Registers are fast per-thread storage. Excessive register use can reduce residency; spills may use local memory, which is backed by device memory and is much slower than registers.

## 10. Global Memory

Global memory is large and high bandwidth but high latency relative to registers/shared memory. Achieved bandwidth depends strongly on access pattern.

## 11. Coalescing

A warp's requests are serviced using memory transactions. Efficient access minimizes transactions and wasted transferred bytes. Contiguous aligned float loads are a common efficient pattern.

Educational segment model:

~~~text
segment = floor(address / transaction_bytes)
~~~

The number of unique segments touched approximates transaction pressure; exact behavior is architecture dependent.

## 12. Shared Memory

Shared memory is block-scoped programmer-managed scratch space on the SM. It is useful for tile reuse and cooperation between threads in one block.

## 13. Synchronization

A block barrier such as __syncthreads() coordinates threads within that block. It is not a general grid-wide barrier.

## 14. Shared-Memory Banks

Shared memory is banked. Different words requested from the same bank by lanes in one warp can serialize, causing bank conflicts.

## 15. Caches

Modern NVIDIA GPUs include per-SM L1/unified cache and a larger device-wide L2 cache. Caches help reuse but should not be expected to rescue systematically poor access patterns.

## 16. Arithmetic Intensity

~~~text
arithmetic_intensity = FLOPs / bytes_moved
~~~

Low intensity often means memory-bound; high intensity can become compute-bound.

## 17. Roofline Intuition

~~~text
achievable FLOP/s
<=
min(peak_compute, memory_bandwidth * arithmetic_intensity)
~~~

This helps decide whether optimization should focus on memory movement or compute throughput.

## 18. Occupancy

Occupancy is constrained by threads, registers, shared memory and architecture limits. High occupancy can help hide latency, but maximum occupancy does not guarantee maximum performance.

## 19. Kernel Launch Overhead

Many tiny GPU operations can lose to launch overhead. This motivates batching, fusion and compiled execution.

## 20. Asynchronous Execution

Many CUDA operations are asynchronous relative to the host. Naive CPU timing can therefore measure only launch overhead. Synchronize or use GPU events when benchmarking completed GPU work.

## 21. Streams and Events

A stream is an ordered queue of GPU work. Independent streams may overlap compute/copy when resources and dependencies permit. CUDA events can timestamp work and express dependencies.

## 22. Transfers and Pinned Memory

CPU↔GPU transfer is not free. Page-locked host memory can support efficient asynchronous transfers in suitable APIs, but pinned memory itself is a finite host resource.

## 23. PyTorch CUDA

Useful APIs include:

~~~python
torch.cuda.is_available()
torch.cuda.get_device_name()
torch.cuda.get_device_properties()
torch.cuda.memory_allocated()
torch.cuda.max_memory_allocated()
torch.cuda.synchronize()
~~~

Guard CUDA-only paths so CPU CI remains runnable.

## 24. Profiling

Measure kernel time, copies, launch count, tensor shapes, synchronizations and memory peaks. Useful tools include PyTorch Profiler, Nsight Systems and Nsight Compute.

## 25. From Scratch

src/cuda_math.py implements:

- ceil_div
- grid_size_1d
- global_thread_index
- warp_and_lane
- transaction_segment_count
- theoretical_resident_blocks
- arithmetic_intensity
- roofline_bound

## 26. Common Mistakes

1. missing tail bounds check
2. confusing grid size with block size
3. ignoring warp divergence
4. uncoalesced global access
5. shared-memory race without synchronization
6. bank conflicts ignored
7. timing asynchronous kernels without synchronization
8. excessive host-device transfers
9. optimizing for occupancy alone
10. custom kernel work before profiling

## 27. Exercises / Mini Project

- [Exercises](exercises/README.md)
- [Solutions](solutions/README.md)
- [Mini Project](mini-project/README.md)

## 28. Checklist

- [ ] kernel / grid / block / thread
- [ ] SM / warp / SIMT
- [ ] divergence
- [ ] registers / global / shared memory
- [ ] coalescing / bank conflicts
- [ ] synchronization
- [ ] streams / events
- [ ] occupancy
- [ ] arithmetic intensity / roofline
- [ ] profiling

## 29. What's Next

Chapter 54 uses lower-precision arithmetic to reduce memory and improve accelerator throughput while preserving training stability.