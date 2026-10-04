# Chapter 53 Exercises

1. Calculate grid size for several N/block sizes.
2. Map block/thread indices to global indices.
3. Identify warp/lane IDs for a 256-thread block.
4. Compare contiguous vs strided transaction segments.
5. Draw register/shared/global memory ownership.
6. Calculate simplified blocks/SM under thread/register/shared limits.
7. Compute arithmetic intensity for vector add and matrix multiply.
8. Apply the roofline bound to a hypothetical GPU.
9. Benchmark a PyTorch CUDA op correctly using synchronization/events.
10. Profile host-device copies vs device-resident computation.

Challenge:
- write and compile a CUDA C++ vector-add kernel
- compare a naive matrix transpose against a tiled shared-memory version
