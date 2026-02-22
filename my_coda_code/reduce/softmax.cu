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

// ========== Softmax Kernel（结合 shfl_xor + 多warp）==========
__global__ void softmax_kernel(const float* __restrict__ x,
                                float* __restrict__ y,
                                const int L) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int warp_num = blockDim.x / 32;

    const float* x_row = x + row * L;
    float* y_row = y + row * L;

    // ===== 第1步：求 max =====
    float local_max = -FLT_MAX;
    for (int i = tid; i < L; i += blockDim.x) {
        local_max = fmaxf(local_max, x_row[i]);
    }

    // warp 内归约（shfl_xor，所有线程都拿到结果）
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }

    // warp 间：每个 warp 的结果存入 shared memory
    __shared__ float s_max[32];
    if (lane == 0) s_max[warp_id] = local_max;
    __syncthreads();

    // 每个线程自己遍历 s_max，拿到全局 max（不需要额外广播）
    float row_max = -FLT_MAX;
    for (int i = 0; i < warp_num; i++) {
        row_max = fmaxf(row_max, s_max[i]);
    }

    // ===== 第2步：求 sum(exp(x - max)) =====
    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        local_sum += expf(x_row[i] - row_max);
    }

    // warp 内归约
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }

    // warp 间
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();

    // 每个线程自己求总和
    float row_sum = 0.0f;
    for (int i = 0; i < warp_num; i++) {
        row_sum += s_sum[i];
    }

    // ===== 第3步：归一化 =====
    for (int i = tid; i < L; i += blockDim.x) {
        y_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}

// ========== CPU 参考 ==========
void cpu_softmax(const float* x, float* y, int batch, int L) {
    for (int row = 0; row < batch; row++) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < L; i++)
            max_val = fmaxf(max_val, x[row * L + i]);

        float sum = 0.0f;
        for (int i = 0; i < L; i++)
            sum += expf(x[row * L + i] - max_val);

        for (int i = 0; i < L; i++)
            y[row * L + i] = expf(x[row * L + i] - max_val) / sum;
    }
}

// ========== 主函数 ==========
int main() {
    const int batch = 64;
    const int L = 4096;
    const int BLOCK_SIZE = 256;

    printf("=== Softmax (shfl_xor + multi-warp) ===\n");
    printf("batch=%d, L=%d, BLOCK_SIZE=%d\n", batch, L, BLOCK_SIZE);
    printf("warp_num=%d, elements_per_thread=%d\n\n", BLOCK_SIZE / 32, L / BLOCK_SIZE);

    size_t bytes = (size_t)batch * L * sizeof(float);

    float* h_x     = (float*)malloc(bytes);
    float* h_y_gpu = (float*)malloc(bytes);
    float* h_y_cpu = (float*)malloc(bytes);

    // 初始化：[-5, 5) 的随机数
    srand(42);
    for (int i = 0; i < batch * L; i++) {
        h_x[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }

    // CPU 参考
    printf("Running CPU softmax...\n");
    cpu_softmax(h_x, h_y_cpu, batch, L);

    // GPU
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, bytes));
    CHECK_CUDA(cudaMalloc(&d_y, bytes));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    // 预热
    softmax_kernel<<<batch, BLOCK_SIZE>>>(d_x, d_y, L);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int nIter = 1000;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) {
        softmax_kernel<<<batch, BLOCK_SIZE>>>(d_x, d_y, L);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float msec = 0;
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, stop));

    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, bytes, cudaMemcpyDeviceToHost));

    // ===== 验证 =====
    printf("\nGPU row0 first 5: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_y_gpu[i]);
    printf("\nCPU row0 first 5: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_y_cpu[i]);
    printf("\n\n");

    // 检查精度
    float max_err = 0;
    for (int i = 0; i < batch * L; i++) {
        float err = fabs(h_y_gpu[i] - h_y_cpu[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max absolute error: %.2e\n", max_err);
    printf("Result: %s\n\n", max_err < 1e-5 ? "PASS ✅" : "FAIL ❌");

    // 检查每行和是否为 1
    printf("Row sums (should be 1.0):\n");
    for (int row = 0; row < 5; row++) {
        float sum = 0;
        for (int i = 0; i < L; i++) sum += h_y_gpu[row * L + i];
        printf("  row %d: %.8f\n", row, sum);
    }
    printf("\n");

    // 性能
    printf("Performance:\n");
    printf("  Time: %.4f ms/iter\n", msec / nIter);
    printf("  Throughput: %.2f GB/s\n",
           2.0 * bytes / (msec / nIter / 1000.0) / 1e9);
    //     ↑ ×2 因为读 x 写 y

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y_gpu);
    free(h_y_cpu);

    return 0;
}