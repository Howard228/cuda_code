#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// dim3 block(BLOCK_SIZE)
// dim3 grid(CEIL(batch, BLOCK_SIZE))
__global__ void cuda_softmax_naive(const float *input, float *output, const int L, const int batch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    input += row * L;
    output += row * L;
    float row_max = -INFINITY;
    for (int i = 0; i < L; ++i) {
        row_max = fmaxf(row_max, input[i]);
    }
    float row_sum = 0.0f;
    for (int i = 0; i < L; ++i) {
        row_sum += expf(input[i] - row_max);
    }
    for (int i = 0; i < L; ++i) {
        output[i] = expf(input[i] - row_max) / row_sum;
    }
}



// dim3 block(BLOCK_SIZE)
// dim3 grid(batch)
__global__ void cuda_softmax(const float *input, float *output, const int L) {
    int row = blockIdx.x;
    input += row * L;
    output += row * L;
    int stride = blockDim.x;
    float row_max = -INFINITY;
    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);
    for (int i = threadIdx.x; i < L; i += stride) {
        row_max = fmaxf(row_max, input[i]);
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, offset));
    }
    __shared__ float s_max[32];
    if (lane == 0) s_max[warp_id] = row_max;
    __syncthreads();
    for (int i = 0; i < warp_num; ++i) {
        row_max = fmaxf(row_max, s_max[i]);
    }
    float row_sum = 0.0f;
    for (int i = threadIdx.x; i < L; i += stride) {
        row_sum += expf(input[i] - row_max);
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, offset);
    }
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp_id] = row_sum;
    __syncthreads();
    row_sum = 0.0f;
    for (int i = 0; i < warp_num; ++i) {
        row_sum += s_sum[i];
    }
    
    for (int i = threadIdx.x; i < L; i += stride) {
        output[i] = expf(input[i] - row_max) / row_sum; 
    }
}