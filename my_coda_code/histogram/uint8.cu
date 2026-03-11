#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ---- Kernel 1: 直方图 ----
__global__ void histogram_kernel(const uint8_t* data, int n, unsigned int* hist) {
    __shared__ unsigned int local_hist[256];
    
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        atomicAdd(&local_hist[data[i]], 1);
    __syncthreads();
    
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        if (local_hist[i] > 0)
            atomicAdd(&hist[i], local_hist[i]);
}

// ---- Kernel 2: 查找第K大 ----
__global__ void find_kth_largest_kernel(const unsigned int* hist, int k, uint8_t* result) {
    unsigned long long cumsum = 0;
    for (int val = 255; val >= 0; val--) {
        cumsum += hist[val];
        if (cumsum >= (unsigned long long)k) {
            *result = (uint8_t)val;
            return;
        }
    }
    *result = 0;
}

// ---- 主机接口 ----
uint8_t find_kth_largest_gpu(const uint8_t* h_data, int n, int k) {
    uint8_t *d_data, *d_result;
    unsigned int* d_hist;
    uint8_t h_result;
    
    cudaMalloc(&d_data, n);
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMalloc(&d_result, 1);
    
    cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));
    
    int block = 256;
    int grid = min((n + block - 1) / block, 1024);
    histogram_kernel<<<grid, block>>>(d_data, n, d_hist);
    find_kth_largest_kernel<<<1, 1>>>(d_hist, k, d_result);
    
    cudaMemcpy(&h_result, d_result, 1, cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_hist);
    cudaFree(d_result);
    return h_result;
}

// ---- 测试 ----
int main() {
    const int N = 1 << 20;  // 1M 个元素
    uint8_t* data = (uint8_t*)malloc(N);
    
    srand(42);
    for (int i = 0; i < N; i++)
        data[i] = rand() % 256;
    
    // 测试几个 K 值
    int test_k[] = {1, 10, 100, 1000, N/2, N};
    
    for (int t = 0; t < 6; t++) {
        int k = test_k[t];
        uint8_t gpu_ans = find_kth_largest_gpu(data, N, k);
        
        // CPU 验证
        unsigned int hist[256] = {};
        for (int i = 0; i < N; i++) hist[data[i]]++;
        unsigned long long cum = 0;
        uint8_t cpu_ans = 0;
        for (int v = 255; v >= 0; v--) {
            cum += hist[v];
            if (cum >= (unsigned long long)k) { cpu_ans = v; break; }
        }
        
        printf("K=%-8d  GPU=%3u  CPU=%3u  %s\n",
               k, gpu_ans, cpu_ans, gpu_ans == cpu_ans ? "✓" : "✗");
    }
    
    free(data);
    return 0;
}