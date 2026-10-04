# Chapter 53 Solutions — Key Ideas

- CUDA work is organized into grids of thread blocks; each thread computes a global index from block/thread coordinates.
- NVIDIA warps contain 32 threads; divergence and memory access are often analyzed at warp granularity.
- coalescing minimizes global-memory transactions, while shared memory enables reusable block-local tiles.
- occupancy is constrained by threads, registers and shared-memory resources but is not itself the final performance objective.
- the roofline model separates memory-bandwidth and compute-throughput limits.
