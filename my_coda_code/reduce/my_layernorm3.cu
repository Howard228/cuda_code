#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// cuda_layernorm_naive<<<CEIL(batch, BLOCK_SIZE), BLOCK_SIZE>>>
__global__ void cuda_layernorm_naive(const float *x, float *y, const float *gramma, const float *beta,
                                     const float eps, const int batch, const int L) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    x += row * L;
    y += row * L;
    float mean = 0.0f;
    for (int i = 0; i < L; ++i) {
        mean += x[i];
    }
    mean /= L;
    float var = 0.0f;
    for (int i = 0; i < L; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= L;
    float inv_std = 1 / sqrtf(var + eps);
    for (int i = 0; i < L; ++i) {
        y[i] = (x[i] - mean) * inv_std * gramma[i] + beta[i];
    }
}


// cuda_layernorm<<<batch, BLOCK_SIZE>>>
__global__ void cuda_layernorm(const float *x, float *y, const float *gramma, const float *beta,
                               const float eps, const int L) {
    int row = blockIdx.x;
    x += row * L;
    y += row * L;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);

    float local_sum = 0.0f;
    for (int i = tid; i < L; i += stride) {
        local_sum += x[i];
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    __shared__ float s_sum[32];
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    float row_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < warp_num; ++i) {
        row_sum += s_sum[i];
    }
    float mean = row_sum / L;

    float local_var = 0.0f;
    for (int i = tid; i < L; i += stride) {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_var += __shfl_xor_sync(0xFFFFFFFF, local_var, offset);
    }
    __shared__ float s_var[32];
    if (lane == 0) {
        s_var[warp_id] = local_var;
    }
    __syncthreads();
    float row_var = 0.0f;
    #pragma unroll
    for (int i = 0; i < warp_num; ++i) {
        row_var += s_var[i];
    }
    row_var /= L;

    float inv_std = rsqrtf(row_var + eps);
    for (int i = tid; i < L; i += stride) {
        y[i] = (x[i] - mean) * inv_std * gramma[i] + beta[i];
    }

}