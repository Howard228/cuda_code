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

// ========== CPU 普通 Softmax ==========
void cpu_softmax(const float *x, float *y, int batch, int L) {
    for (int row = 0; row < batch; row++) {
        const float *x_row = x + row * L;
        float *y_row = y + row * L;
        float row_max = -INFINITY;
        for (int i = 0; i < L; i++)
            row_max = fmaxf(row_max, x_row[i]);
        float row_sum = 0.0f;
        for (int i = 0; i < L; i++)
            row_sum += expf(x_row[i] - row_max);
        for (int i = 0; i < L; i++)
            y_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}

// ========== CPU Online Softmax ==========
void cpu_online_softmax(const float *x, float *y, int batch, int L) {
    for (int row = 0; row < batch; row++) {
        const float *x_row = x + row * L;
        float *y_row = y + row * L;

        // 第1遍：同时求 max 和 sum
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        for (int i = 0; i < L; i++) {
            float old_max = row_max;
            row_max = fmaxf(row_max, x_row[i]);
            row_sum = row_sum * expf(old_max - row_max) + expf(x_row[i] - row_max);
        }

        // 第2遍：归一化
        for (int i = 0; i < L; i++) {
            y_row[i] = expf(x_row[i] - row_max) / row_sum;
        }
    }
}

// ========== CUDA Online Softmax ==========
// 每行一个 block，<<<batch, BLOCK_SIZE>>>
__global__ void cuda_online_softmax(const float *x, float *y, const int L) {
    const int WARP_SIZE = 32;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warp_num = blockDim.x / WARP_SIZE;

    const float *x_row = x + row * L;
    float *y_row = y + row * L;

    // ===== 第1遍：每线程 online 求局部 max 和 sum =====
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        float val = x_row[i];
        float old_max = local_max;
        local_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(old_max - local_max) + expf(val - local_max);
    }

    // ===== warp 内归约 max 和 sum =====
    // 两个线程合并：需要同时归约 max 和 sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        float other_sum = __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        // 合并两个 (max, sum) 对
        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - new_max) 
                  + other_sum * expf(other_max - new_max);
        local_max = new_max;
    }

    // ===== warp 间归约 =====
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
    if (lane == 0) {
        s_max[warp_id] = local_max;
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // 每线程遍历所有 warp 的结果，做 online 合并
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    for (int i = 0; i < warp_num; i++) {
        float other_max = s_max[i];
        float other_sum = s_sum[i];
        float new_max = fmaxf(row_max, other_max);
        row_sum = row_sum * expf(row_max - new_max)
                + other_sum * expf(other_max - new_max);
        row_max = new_max;
    }

    // ===== 第2遍：归一化 =====
    for (int i = tid; i < L; i += blockDim.x) {
        y_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}

// ========== CUDA 普通 Softmax（对比用）==========
__global__ void cuda_normal_softmax(const float *x, float *y, const int L) {
    const int WARP_SIZE = 32;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warp_num = blockDim.x / WARP_SIZE;
    const float *x_row = x + row * L;
    float *y_row = y + row * L;

    // 第1遍：max
    float local_max = -INFINITY;
    for (int i = tid; i < L; i += blockDim.x)
        local_max = fmaxf(local_max, x_row[i]);
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    __shared__ float s_max2[32];
    if (lane == 0) s_max2[warp_id] = local_max;
    __syncthreads();
    float row_max = -INFINITY;
    for (int i = 0; i < warp_num; i++)
        row_max = fmaxf(row_max, s_max2[i]);

    // 第2遍：sum
    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x)
        local_sum += expf(x_row[i] - row_max);
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    __shared__ float s_sum2[32];
    if (lane == 0) s_sum2[warp_id] = local_sum;
    __syncthreads();
    float row_sum = 0.0f;
    for (int i = 0; i < warp_num; i++)
        row_sum += s_sum2[i];

    // 第3遍：归一化
    for (int i = tid; i < L; i += blockDim.x)
        y_row[i] = expf(x_row[i] - row_max) / row_sum;
}

// ========== 主函数 ==========
int main() {
    const int batch = 64;
    const int L = 4096;
    const int BLOCK_SIZE = 256;

    printf("=== Online Softmax vs Normal Softmax ===\n");
    printf("batch=%d, L=%d\n\n", batch, L);

    size_t bytes = (size_t)batch * L * sizeof(float);
    float *h_x      = (float *)malloc(bytes);
    float *h_y_cpu   = (float *)malloc(bytes);
    float *h_y_online = (float *)malloc(bytes);
    float *h_y_normal = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < batch * L; i++)
        h_x[i] = (float)(rand() % 1000) / 100.0f - 5.0f;

    // CPU
    cpu_softmax(h_x, h_y_cpu, batch, L);

    // GPU
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, bytes));
    CHECK_CUDA(cudaMalloc(&d_y, bytes));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float msec;
    int nIter = 1000;

    // ===== 普通 Softmax =====
    cuda_normal_softmax<<<batch, BLOCK_SIZE>>>(d_x, d_y, L);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
        cuda_normal_softmax<<<batch, BLOCK_SIZE>>>(d_x, d_y, L);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, stop));
    CHECK_CUDA(cudaMemcpy(h_y_normal, d_y, bytes, cudaMemcpyDeviceToHost));
    float time_normal = msec / nIter;

    // ===== Online Softmax =====
    cuda_online_softmax<<<batch, BLOCK_SIZE>>>(d_x, d_y, L);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
        cuda_online_softmax<<<batch, BLOCK_SIZE>>>(d_x, d_y, L);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, stop));
    CHECK_CUDA(cudaMemcpy(h_y_online, d_y, bytes, cudaMemcpyDeviceToHost));
    float time_online = msec / nIter;

    // ===== 验证 =====
    printf("CPU first 5:     ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_y_cpu[i]);
    printf("\nNormal first 5:  ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_y_normal[i]);
    printf("\nOnline first 5:  ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_y_online[i]);
    printf("\n\n");

    float max_err_n = 0, max_err_o = 0;
    for (int i = 0; i < batch * L; i++) {
        float err_n = fabs(h_y_normal[i] - h_y_cpu[i]);
        float err_o = fabs(h_y_online[i] - h_y_cpu[i]);
        if (err_n > max_err_n) max_err_n = err_n;
        if (err_o > max_err_o) max_err_o = err_o;
    }
    printf("Normal error: %.2e %s\n", max_err_n, max_err_n < 1e-5 ? "✅" : "❌");
    printf("Online error: %.2e %s\n\n", max_err_o, max_err_o < 1e-5 ? "✅" : "❌");

    printf("Normal time: %.4f ms\n", time_normal);
    printf("Online time: %.4f ms\n", time_online);
    printf("Speedup:     %.2fx\n", time_normal / time_online);

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y_cpu); free(h_y_normal); free(h_y_online);
    return 0;
}
