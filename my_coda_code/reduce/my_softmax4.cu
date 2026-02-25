#include <cmath>
#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// 调用方式：cuda_softmax_naive<<<CEIL(batch, 256), 256>>>(x, y, batch, L);每个线程处理一整行
__global__ void cuda_softmax_naive(float *x, float *y, const int batch, const int L) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;

    float *x_row = x + row * L;
    float *y_row = y + row * L;

    // 1. 找最大值
    float max_val = -INFINITY;
    for (int i = 0; i < L; ++i) {
        max_val = fmaxf(max_val, x_row[i]);
    }

    // 2. 求 exp 之和
    float sum = 0.0f;
    for (int i = 0; i < L; ++i) {
        sum += expf(x_row[i] - max_val);
    }

    // 3. 归一化
    for (int i = 0; i < L; ++i) {
        y_row[i] = expf(x_row[i] - max_val) / sum;
    }
}

// cuda_softmax_reduce<<<batch, BLOCK_SIZE>>>
__global__ void cuda_softmax_reduce(float *x, float *y, const int L) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);
    x += row * L;
    y += row * L;
    float local_max = -INFINITY;
    for (int i = tid; i < L; i += stride) {
        local_max = fmaxf(local_max, x[i]);
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));   
    }
    __shared__ float s_max[32];
    if (lane == 0) s_max[warp_id] = local_max;
    __syncthreads();
    float row_max = -INFINITY;
    for (int i = 0; i < warp_num; ++i) {
        row_max = fmaxf(row_max, s_max[i]);
    }

    float local_sum = 0.0f;
    for (int i = tid; i < L; i += stride) {
        local_sum += expf(x[i] - row_max);
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

    for (int i = tid; i < L; i += stride) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}