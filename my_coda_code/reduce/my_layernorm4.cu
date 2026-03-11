#include <__clang_cuda_math.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// cuda_layernorm_naive<<<CEIL(batch, BLOCK_SIZE), BLOCK_SIZE>>>
__global__ void cuda_layernorm_naive(const float *input, float *output, const float *gramma, const float *beta, 
                                     const int L, const float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    input += row * L;
    output += row * L;
    float mean = 0.0f;
    for (int i = 0; i < L; ++i) {
        mean += input[i];
    }
    mean /= L;
    float sum = 0.0f;
    for (int i = 0; i < L; ++i) {
        float diff = input[i] - mean;
        sum += diff * diff;
    }
    sum /= L;
    float inv_std = 1 / sqrtf(sum + eps);
    for (int i = 0; i < L; ++i) {
        output[i] = (input[i] - mean) * inv_std * gramma[i] + beta[i];
    }
}

// cuda_layernorm<<<batch, BLOCK_SIZE>>> 
__global__ void cuda_layernorm(const float *input, float *output, const float *gramma, const float *beta, 
                               const int L, const float eps) {
    const int row = blockIdx.x;
    input += row * L;
    output += row * L;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);

    float local_mean = 0.0f;
    for (int i = tid; i < L; i += stride) {
        local_mean += input[i];
    }
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_mean += __shfl_xor_sync(0xFFFFFFFF, local_mean, offset);
    }
    __shared__ float s_mean[32];
    if (lane == 0) s_mean[warp_id] = local_mean;
    __syncthreads();
    float row_mean = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        row_mean += s_mean[i];
    }
    row_mean /= L;

    float local_var = 0.0f;
    for (int i = tid; i < L; i += stride) {
        float diff = input[i] - row_mean;
        local_var += diff * diff;
    }
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
    for (int i = tid; i < L; i += stride) {
        output[i] = (input[i] - row_mean) * inv_std * gramma[i] + beta[i];
    }
}
