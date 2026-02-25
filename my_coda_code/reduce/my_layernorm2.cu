#include <cmath>
#include <cuda_runtime.h>

void cpu_layernorm(const float* __restrict__ x, float* __restrict__ y,
                   const float* __restrict__ gramma, const float* __restrict__ beta,
                   const int batch, const int L, float eps) {
    for (int row = 0; row < batch; ++row) {
        const float* x_row = x + row * L;
        float* y_row = y + row * L;
        float mean = 0.0f;
        for (int col = 0; col < L; ++col) {
            mean += x_row[col];
        }
        mean /= L;
        float var = 0.0f;
        for (int col = 0; col < L; ++col) {
            float diff = x_row[col] - mean;
            var += diff * diff;
        }
        var /= L;
        var = 1 / sqrtf(var + eps);
        for (int col = 0; col < L; ++col) {
            y_row[col] = (x_row[col] - mean) * var * gramma[col] + beta[col];
        }
    }
}

// cuda_layernorm<<<batch, BLOCK_SIZE>>>
__global__ void cuda_layernorm(const float* __restrict__ x, float* __restrict__ y,
                               const float* __restrict__ gramma, const float* __restrict__ beta,
                               const int L, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    const int WARP_SIZE = 32;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_num = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    const float* x_row = x + row * L;
    float* y_row = y + row * L;
    float local_sum = 0.0f;
    for (int i = tid; i < L; i += stride) {
        local_sum += x_row[i];
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        sum += s_sum[i];
    }
    float mean = sum / L;

    float local_var_sum = 0.0f;
    for (int i = tid; i < L; i += stride) {
        float diff = x_row[i] - mean;
        local_var_sum += diff * diff;
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_var_sum += __shfl_xor_sync(0xFFFFFFFF, local_var_sum, offset);
    }
    __shared__ float s_var_sum[32];
    if (lane == 0) s_var_sum[warp_id] = local_var_sum;
    __syncthreads();
    float var_sum = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        var_sum += s_var_sum[i];
    }
    var_sum /= L;
    float inv_std = rsqrtf(var_sum + eps);

    for (int i = tid; i < L; i += stride) {
        y_row[i] = (x_row[i] - mean) * inv_std * gramma[i] + beta[i];
    }
}