#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// cuda_layernorm_naive<<<CEIL(batch, BLOCK_SIZE), BLOCK_SIZE>>>
__global__ void cuda_layernorm_naive(const float *input, float *output, const float *gramma, 
                                     const float *beta, const int L, const float eps, const int batch) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    input += row * L;
    output += row * L;
    float mean = 0.0f;
    for (int i = 0; i < L; ++i) {
        mean += input[i];
    }
    mean /= L;
    float var = 0.0f;
    for (int i = 0; i < L; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= L;
    float inv_std = 1 / sqrtf(var + eps);
    for (int i = 0; i < L; ++i) {
        output[i] = (input[i] - mean) * inv_std * gramma[i] + beta[i];
    }
}



// cuda_layernorm<<<batch, BLOCK_SIZE>>>
__global__ void cuda_layernorm(const float *input, float *output, const float *gramma, 
                                     const float *beta, const int L, const float eps) {
    int row = blockIdx.x;
    input += row * L;
    output += row * L;
    int stride = blockDim.x;
    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);

    float row_mean = 0.0f;
    for (int i = threadIdx.x; i < L; i += stride) {
        row_mean += input[i];
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        row_mean += __shfl_xor_sync(0xFFFFFFFF, row_mean, offset);
    }
    __shared__ float s_mean[32];
    if (lane == 0) s_mean[warp_id] = row_mean;
    __syncthreads();
    row_mean = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        row_mean += s_mean[i];
    }
    row_mean /= L;
    float row_var = 0.0f;
    for (int i = threadIdx.x; i < L; i += stride) {
        float diff = input[i] - row_mean;
        row_var += diff * diff;
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        row_var += __shfl_xor_sync(0xFFFFFFFF, row_var, offset);
    }
    __shared__ float s_var[32];
    if (lane == 0) s_var[warp_id] = row_var;
    __syncthreads();
    row_var = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        row_var += s_var[i];
    }
    row_var /= L;
    float inv_std = rsqrtf(row_var + eps);
    for (int i = threadIdx.x; i < L; i += stride) {
        output[i] = (input[i] - row_mean) * inv_std * gramma[i] + beta[i];
    }

}