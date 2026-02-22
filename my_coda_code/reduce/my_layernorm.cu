#include <cmath>
#include <cuda_runtime.h>

void cpu_layernorm(const float *x, float *y, 
                   float *gamma, float *beta, 
                   const int batch, const int L,
                   const float eps) {
    for (int row = 0; row < batch; ++row) {
        const float *x_row = x + row * L;
        float *y_row = y + row * L;
        float mean = 0.0f;
        for (int i = 0; i < L; ++i) {
            mean += x_row[i];
        }
        mean /= L;
        float var = 0.0f;
        for (int i = 0; i < L; ++i) {
            float diff = x_row[i] - mean;
            var += diff * diff;
        }
        var /= L;
        float inv_std = 1 / sqrtf(var + eps);
        for (int i = 0; i < L; ++i) {
            y_row[i] = (x_row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

// cuda_layernorm<<<batch, BLOCK_SIZE>>>
__global__ void cuda_layernorm(const float *x, float *y, 
                               float *gamma, float *beta, 
                               const int L, const float eps) {
    int row = blockIdx.x;
    const int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_num = blockDim.x / WARP_SIZE;
    const float *x_row = x + row * L;
    float *y_row = y + row * L;
    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        local_sum += x_row[i];
    }
    #pragma unroll 
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();
    float row_sum = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        row_sum += s_sum[i];
    }
    float mean = row_sum / L;

    float local_var = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        float diff = (x_row[i] - mean);
        local_var += diff * diff;
    }
    #pragma unroll 
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_var += __shfl_xor_sync(0xFFFFFFFF, local_var, offset);
    }
    __shared__ float s_var[32];
    if (lane == 0) s_var[warp_id] = local_var;
    __syncthreads();
    float row_var = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        row_var += s_var[i];
    }
    row_var /= L;
    float inv_std = rsqrtf(row_var + eps);
    for (int i = tid; i < L; i += blockDim.x) {
        y_row[i] = (x_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}