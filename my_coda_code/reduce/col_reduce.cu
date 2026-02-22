#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                         \
            exit(1);                                                \
        }                                                           \
    } while (0)

// ============================================================
// ❌ 方式一：竖着读（每 block 处理一列）
// ============================================================
__global__ void col_reduce_vertical(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     const int batch, const int L) {
    int col = blockIdx.x;   // 每个 block 负责一列
    int tid = threadIdx.x;
    if (col >= L) return;

    float sum = 0.0f;
    for (int row = tid; row < batch; row += blockDim.x) {
        sum += x[row * L + col];
        // warp 内 32 个线程读的是：
        //   x[tid*L + col], x[(tid+1)*L + col], x[(tid+2)*L + col], ...
        //   间隔 L*4 = 512 字节
        //   每个线程触发一个 cache line → 32 个 cache line → 浪费 ❌
    }

    // block 内归约（简化用 atomicAdd）
    atomicAdd(&y[col], sum);
}

// ============================================================
// ✅ 方式二：横着读（每线程一列，一个 warp 读一行）
// ============================================================
__global__ void col_reduce_horizontal(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       const int batch, const int L) {
    int col = threadIdx.x;  // 线程0→col0, 线程1→col1, ...
    if (col >= L) return;

    float sum = 0.0f;
    for (int row = blockIdx.x; row < batch; row += gridDim.x) {
        sum += x[row * L + col];
        // warp 内 32 个线程读的是：
        //   x[row*L + 0], x[row*L + 1], x[row*L + 2], ...
        //   连续地址！一个 cache line 搞定 → 合并访存 ✅
    }

    atomicAdd(&y[col], sum);
}

// ============================================================
// CPU 参考
// ============================================================
void cpu_col_reduce(const float* x, float* y, int batch, int L) {
    for (int j = 0; j < L; j++) y[j] = 0.0f;
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < L; j++) {
            y[j] += x[i * L + j];
        }
    }
}

int main() {
    const int batch = 1000000;
    const int L = 128;

    printf("=== Column Reduce: (%d, %d) → (1, %d) ===\n\n", batch, L, L);

    size_t bytes_x = (size_t)batch * L * sizeof(float);
    size_t bytes_y = L * sizeof(float);

    float* h_x     = (float*)malloc(bytes_x);
    float* h_y_v   = (float*)malloc(bytes_y);  // 竖着读的结果
    float* h_y_h   = (float*)malloc(bytes_y);  // 横着读的结果
    float* h_y_cpu = (float*)malloc(bytes_y);  // CPU 结果

    srand(42);
    for (long long i = 0; i < (long long)batch * L; i++)
        h_x[i] = (float)(rand() % 100) / 100.0f;

    // CPU
    printf("CPU computing...\n");
    cpu_col_reduce(h_x, h_y_cpu, batch, L);

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, bytes_x));
    CHECK_CUDA(cudaMalloc(&d_y, bytes_y));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float msec;
    int nIter = 100;

    // ============================================================
    // 测试方式一：竖着读 ❌
    // ============================================================
    CHECK_CUDA(cudaMemset(d_y, 0, bytes_y));
    col_reduce_vertical<<<L, 256>>>(d_x, d_y, batch, L);  // 128 个 block
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) {
        CHECK_CUDA(cudaMemset(d_y, 0, bytes_y));
        col_reduce_vertical<<<L, 256>>>(d_x, d_y, batch, L);
        //                   ↑
        //               128 个 block，每 block 一列
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, stop));
    CHECK_CUDA(cudaMemcpy(h_y_v, d_y, bytes_y, cudaMemcpyDeviceToHost));

    float time_v = msec / nIter;
    float bw_v = (double)batch * L * sizeof(float) / (time_v / 1000.0) / 1e9;
    printf("竖着读: %.3f ms, %.2f GB/s\n", time_v, bw_v);

    // ============================================================
    // 测试方式二：横着读 ✅
    // ============================================================
    CHECK_CUDA(cudaMemset(d_y, 0, bytes_y));
    // 一个block负责多行的规约，每个thread读取时读取一列
    col_reduce_horizontal<<<256, L>>>(d_x, d_y, batch, L);  // 256 个 block
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) {
        CHECK_CUDA(cudaMemset(d_y, 0, bytes_y));
        col_reduce_horizontal<<<256, L>>>(d_x, d_y, batch, L);
        //                               ↑             ↑
        //                           256个block  128线程/block
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, stop));
    CHECK_CUDA(cudaMemcpy(h_y_h, d_y, bytes_y, cudaMemcpyDeviceToHost));

    float time_h = msec / nIter;
    float bw_h = (double)batch * L * sizeof(float) / (time_h / 1000.0) / 1e9;
    printf("横着读: %.3f ms, %.2f GB/s\n\n", time_h, bw_h);

    // ============================================================
    // 验证
    // ============================================================
    printf("CPU first 5: ");
    for (int i = 0; i < 5; i++) printf("%.2f ", h_y_cpu[i]);

    printf("\n竖着读 first 5: ");
    for (int i = 0; i < 5; i++) printf("%.2f ", h_y_v[i]);

    printf("\n横着读 first 5: ");
    for (int i = 0; i < 5; i++) printf("%.2f ", h_y_h[i]);
    printf("\n\n");

    // 验证正确性
    float max_err_v = 0, max_err_h = 0;
    for (int i = 0; i < L; i++) {
        float err_v = fabs(h_y_v[i] - h_y_cpu[i]) / fabs(h_y_cpu[i]);
        float err_h = fabs(h_y_h[i] - h_y_cpu[i]) / fabs(h_y_cpu[i]);
        if (err_v > max_err_v) max_err_v = err_v;
        if (err_h > max_err_h) max_err_h = err_h;
    }
    printf("竖着读 error: %.2e %s\n", max_err_v, max_err_v < 1e-3 ? "✅" : "❌");
    printf("横着读 error: %.2e %s\n\n", max_err_h, max_err_h < 1e-3 ? "✅" : "❌");

    printf("加速比: %.2fx\n", time_v / time_h);

    // 清理
    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y_v); free(h_y_h); free(h_y_cpu);
    return 0;
}