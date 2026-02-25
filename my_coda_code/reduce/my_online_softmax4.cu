#include <cmath>
#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// cuda_online_softmax_naive(CEIL(batch, 256), 256) 每个线程处理一行
__global__ void cuda_online_softmax_naive(float *x, float *y, const int batch, const int L) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    x += row * L;
    y += row * L;
    float row_sum = 0.0f;
    float row_max = -INFINITY;
    for (int i = 0; i < L; ++i) {
        float old_max = row_max;
        row_max = fmaxf(row_max, x[i]);
        row_sum = row_sum * expf(old_max - row_max) + expf(x[i] - row_max);
    }
    for (int i = 0; i < L; ++i) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }
}


// cuda_online_softmax<<<batch, BLOCK_SIZE>>>
__global__ void cuda_online_softmax(const float *x, float *y, const int L) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);

    float local_max = -INFINITY;
    float local_sum = 0.0f;

    x += row * L;
    y += row * L;

    for (int i = tid; i < L; i += stride) {
        float old_max = local_max;
        local_max = fmaxf(local_max, x[i]);
        local_sum = local_sum * expf(old_max - local_max) + expf(x[i] - local_max);
    }

    __shared__ float s_sum[32];
    __shared__ float s_max[32];

    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        float other_sum = __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        float new_max = fmaxf(other_max, local_max);
        local_sum = other_sum * expf(other_max - new_max) + local_sum * expf(local_max - new_max);
        local_max = new_max;
    }

    if (lane == 0) {
        s_max[warp_id] = local_max;
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (int i = 0; i < warp_num; ++i) {
        float old_max = row_max;
        row_max = fmaxf(row_max, s_max[i]);
        row_sum = row_sum * expf(old_max - row_max) + s_sum[i] * expf(s_max[i] - row_max);
    }

    for (int i = tid; i < L; i += stride) {
        y[i] = expf(x[i] - row_max) / row_sum;
    }

}