#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                         \
            exit(1);                                                \
        }                                                           \
    } while (0)

// ========== block 内 max 归约 ==========
__inline__ __device__ float block_reduce_max(float val) {
    const int warp_size = 32;
    int lane = threadIdx.x % warp_size;
    int warp_id = threadIdx.x / warp_size;

    // warp 内 shuffle 归约：+ 换成 max
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        //  ↑ 这里从 += 改成 max

    __shared__ float warp_maxs[32];
    if (lane == 0) warp_maxs[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (threadIdx.x < blockDim.x / warp_size) ? warp_maxs[threadIdx.x] : -FLT_MAX;
        //                                                                       ↑ 填负无穷，不是0
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2)
            val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float atomic_max_float(float *addr, float value) {
    int *old_addr = reinterpret_cast<int*>(addr);
    int old = *(old_addr);
    int expected;
    do {
        expected = old;
        // old 返回的是 addr 里的【原始值】，不是 new_val
        old = atomicCAS(old_addr, expected, __float_as_int(max(__int_as_float(expected), value)));
    } while(expected != old);
    return __int_as_float(old); // 返回的是【原始值】
}

// ========== Kernel ==========
__global__ void row_reduce_max(const float* __restrict__ x,
                                float* __restrict__ y,
                                const int L,
                                const int blocks_per_row) {
    int row = blockIdx.x / blocks_per_row;
    int block_in_row = blockIdx.x % blocks_per_row;
    int tid = threadIdx.x;
    int global_tid = block_in_row * blockDim.x + tid;
    int stride = blocks_per_row * blockDim.x;

    float local_max = -FLT_MAX;  // ★ 初始值是负无穷，不是0 ★

    // float4 向量化加载
    int L4 = L / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x + row * L);
    for (int i = global_tid; i < L4; i += stride) {
        float4 v = x4[i];
        local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
        //         ↑ 四个值取最大
    }

    // 尾部处理
    int tail_start = L4 * 4;
    for (int i = tail_start + global_tid; i < L; i += stride) {
        local_max = max(local_max, x[row * L + i]);
    }

    // block 内归约
    local_max = block_reduce_max(local_max);

    // 原子取最大值
    if (tid == 0) {
        atomic_max_float(&y[row], local_max);
        // ↑ 不是 atomicAdd，是 atomicMax
    }
}

// ========== CPU 参考 ==========
void cpu_row_max(const float* x, float* y, int batch, int L) {
    for (int row = 0; row < batch; row++) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < L; i++) {
            if (x[row * L + i] > max_val)
                max_val = x[row * L + i];
        }
        y[row] = max_val;
    }
}

// ========== 初始化 y 为负无穷 ==========
__global__ void init_neg_inf(float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = -FLT_MAX;
}

// ========== 主函数 ==========
int main() {
    const int batch = 64;
    const int L = 10000000;
    const int BLOCK_SIZE = 256;
    const int blocks_per_row = 128;
    const int total_blocks = batch * blocks_per_row;

    printf("=== Row Reduce Max ===\n");
    printf("batch=%d, L=%dM\n\n", batch, L / 1000000);

    size_t bytes_x = (size_t)batch * L * sizeof(float);
    size_t bytes_y = batch * sizeof(float);

    float* h_x = (float*)malloc(bytes_x);
    float* h_y_gpu = (float*)malloc(bytes_y);
    float* h_y_cpu = (float*)malloc(bytes_y);

    srand(42);
    for (long long i = 0; i < (long long)batch * L; i++)
        h_x[i] = (float)(rand() % 10000) / 100.0f;  // [0, 100)

    // CPU
    cpu_row_max(h_x, h_y_cpu, batch, L);
    printf("CPU first 5: ");
    for (int i = 0; i < 5; i++) printf("%.2f ", h_y_cpu[i]);
    printf("\n");

    // GPU
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, bytes_x));
    CHECK_CUDA(cudaMalloc(&d_y, bytes_y));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));

    // ★ y 初始化为负无穷（不能用 cudaMemset，因为 -FLT_MAX 不是全0）★
    init_neg_inf<<<(batch + 255) / 256, 256>>>(d_y, batch);

    row_reduce_max<<<total_blocks, BLOCK_SIZE>>>(d_x, d_y, L, blocks_per_row);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, bytes_y, cudaMemcpyDeviceToHost));

    printf("GPU first 5: ");
    for (int i = 0; i < 5; i++) printf("%.2f ", h_y_gpu[i]);
    printf("\n\n");

    // 验证
    bool correct = true;
    for (int i = 0; i < batch; i++) {
        if (fabs(h_y_gpu[i] - h_y_cpu[i]) > 1e-3) {
            printf("MISMATCH row %d: GPU=%.4f CPU=%.4f\n", i, h_y_gpu[i], h_y_cpu[i]);
            correct = false;
        }
    }
    printf("%s\n", correct ? "PASS ✅" : "FAIL ❌");

    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y_gpu);
    free(h_y_cpu);
    return 0;
}