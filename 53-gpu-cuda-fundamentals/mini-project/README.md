# Mini Project — GPU Performance Lab

On an NVIDIA CUDA machine:

1. inspect torch.cuda device properties
2. benchmark CPU vs GPU vector operations
3. measure H2D/D2H transfer time
4. benchmark several matrix sizes
5. compare synchronized wall-clock vs CUDA-event timing
6. profile a workload
7. calculate arithmetic intensity
8. identify memory-bound vs compute-bound regions

Optional:
- implement vector add and tiled transpose in CUDA C++
- compare transaction/coalescing behavior with Nsight Compute
