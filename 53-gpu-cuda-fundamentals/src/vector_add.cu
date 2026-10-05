// Educational CUDA vector addition.
// Compile only on a system with the NVIDIA CUDA toolkit installed.
// Example: nvcc vector_add.cu -O2 -o vector_add

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1024;
    const size_t bytes = n * sizeof(float);
    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n, 0.0f);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    std::printf("c[0]=%.1f c[last]=%.1f\n", h_c.front(), h_c.back());

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return (h_c.front() == 3.0f && h_c.back() == 3.0f) ? 0 : 1;
}
