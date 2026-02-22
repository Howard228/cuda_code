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

// ========== CPU 参考 ==========
void cpu_layernorm(const float *x, float *y,
                   const float *gamma, const float *beta,
                   const int batch, const int L, const float eps) {
    for (int row = 0; row < batch; ++row) {
        const float *x_row = x + row * L;
        float *y_row = y + row * L;

        // 第1步：求 mean
        float mean = 0.0f;
        for (int i = 0; i < L; ++i) {
            mean += x_row[i];
        }
        mean /= L;

        // 第2步：求 var
        float var = 0.0f;
        for (int i = 0; i < L; ++i) {
            float diff = x_row[i] - mean;
            var += diff * diff;
        }
        var /= L;

        // 第3步：归一化
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < L; ++i) {
            y_row[i] = (x_row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

// ========== GPU Kernel ==========
// cuda_layernorm<<<batch, BLOCK_SIZE>>>
__global__ void cuda_layernorm(const float *x, float *y,
                                const float *gamma, const float *beta,
                                const int L, const float eps) {
    const int WARP_SIZE = 32;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warp_num = blockDim.x / WARP_SIZE;

    const float *x_row = x + row * L;
    float *y_row = y + row * L;

    // ===== 第1步：求 mean（reduce_sum） =====
    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        local_sum += x_row[i];
    }

    // warp 内归约
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }

    // warp 间归约
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();

    float mean = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        mean += s_sum[i];
    }
    mean /= L;

    // ===== 第2步：求 var（reduce_sum） =====
    float local_var = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        float diff = x_row[i] - mean;
        local_var += diff * diff;
    }

    // warp 内归约
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_var += __shfl_xor_sync(0xFFFFFFFF, local_var, offset);
    }

    // warp 间归约
    __shared__ float s_var[32];
    if (lane == 0) s_var[warp_id] = local_var;
    __syncthreads();

    float var = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        var += s_var[i];
    }
    var /= L;

    // ===== 第3步：归一化 =====
    float inv_std = rsqrtf(var + eps);  // rsqrtf = 1/sqrt，比除法快
    for (int i = tid; i < L; i += blockDim.x) {
        y_row[i] = (x_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ========== 主函数 ==========
int main() {
    const int batch = 64;
    const int L = 4096;
    const int BLOCK_SIZE = 256;
    const float eps = 1e-5f;

    printf("=== LayerNorm ===\n");
    printf("batch=%d, L=%d\n\n", batch, L);

    size_t bytes_x = (size_t)batch * L * sizeof(float);
    size_t bytes_param = L * sizeof(float);

    float *h_x       = (float *)malloc(bytes_x);
    float *h_y_gpu   = (float *)malloc(bytes_x);
    float *h_y_cpu   = (float *)malloc(bytes_x);
    float *h_gamma   = (float *)malloc(bytes_param);
    float *h_beta    = (float *)malloc(bytes_param);

    // 初始化
    srand(42);
    for (int i = 0; i < batch * L; i++)
        h_x[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    for (int i = 0; i < L; i++) {
        h_gamma[i] = 1.0f;   // 常见初始化
        h_beta[i] = 0.0f;
    }

    // CPU
    printf("Running CPU layernorm...\n");
    cpu_layernorm(h_x, h_y_cpu, h_gamma, h_beta, batch, L, eps);

    // GPU
    float *d_x, *d_y, *d_gamma, *d_beta;
    CHECK_CUDA(cudaMalloc(&d_x, bytes_x));
    CHECK_CUDA(cudaMalloc(&d_y, bytes_x));
    CHECK_CUDA(cudaMalloc(&d_gamma, bytes_param));
    CHECK_CUDA(cudaMalloc(&d_beta, bytes_param));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, bytes_param, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta, bytes_param, cudaMemcpyHostToDevice));

    // 预热
    cuda_layernorm<<<batch, BLOCK_SIZE>>>(d_x, d_y, d_gamma, d_beta, L, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int nIter = 1000;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) {
        cuda_layernorm<<<batch, BLOCK_SIZE>>>(d_x, d_y, d_gamma, d_beta, L, eps);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float msec = 0;
    CHECK_CUDA(cudaEventElapsedTime(&msec, start, stop));

    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, bytes_x, cudaMemcpyDeviceToHost));

    // 验证
    printf("\nGPU row0 first 5: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_y_gpu[i]);
    printf("\nCPU row0 first 5: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_y_cpu[i]);
    printf("\n\n");

    float max_err = 0;
    for (int i = 0; i < batch * L; i++) {
        float err = fabs(h_y_gpu[i] - h_y_cpu[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max absolute error: %.2e\n", max_err);
    printf("Result: %s\n\n", max_err < 1e-4 ? "PASS ✅" : "FAIL ❌");

    // 验证均值≈0，方差≈1
    printf("After LayerNorm (should be mean≈0, std≈1):\n");
    for (int row = 0; row < 3; row++) {
        float mean = 0, var = 0;
        for (int i = 0; i < L; i++) mean += h_y_gpu[row * L + i];
        mean /= L;
        for (int i = 0; i < L; i++) {
            float d = h_y_gpu[row * L + i] - mean;
            var += d * d;
        }
        var /= L;
        printf("  row %d: mean=%.6f, std=%.6f\n", row, mean, sqrtf(var));
    }
    printf("\n");

    printf("Time: %.4f ms/iter\n", msec / nIter);

    // 清理
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_x); free(h_y_gpu); free(h_y_cpu); free(h_gamma); free(h_beta);
    return 0;
}