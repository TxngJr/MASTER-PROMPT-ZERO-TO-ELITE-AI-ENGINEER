# CUDA Lab — Vector Addition Kernel

This optional lab runs only when a compatible NVIDIA driver and CUDA toolkit with `nvcc` are installed. The chapter remains learnable on CPU without it.

## Kernel mapping

For vectors of length `n`:

~~~text
global thread index =
blockIdx.x * blockDim.x + threadIdx.x
~~~

Each valid thread computes one output element.

## Build

~~~bash
nvcc -O2 src/vector_add.cu -o /tmp/vector_add
/tmp/vector_add
~~~

The program checks CUDA API errors and validates GPU output against a CPU reference.

## Experiments

1. Change `n` from tiny to large.
2. Try block sizes 64, 128, 256 and 512 where supported.
3. Explain why launch configuration affects utilization but not the mathematical answer.
4. Add an intentional missing bounds check on a non-multiple-of-block-size vector in a safe throwaway branch, reason about why it is invalid, then restore the check before execution.
5. Time copies separately from the kernel; do not claim kernel-only speed as end-to-end application speed.
