#include <cmath>
#include <cuda_runtime.h>

void cpu_online_softmax(const float* __restrict__ x, float* __restrict__ y,
                        const int batch, const int L) {
    for (int row = 0; row < batch; ++row) {
        const float *x_row = x + row * L;
        float *y_row = y + row * L;
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        for (int i = 0; i < L; ++i) {
            float old_max = row_max;
            row_max = fmaxf(row_max, x_row[i]);
            row_sum = row_sum * expf(old_max - row_max) + expf(x_row[i] - row_max);
        }
        for (int i = 0; i < L; ++i) {
            y_row[i] = expf(x_row[i] - row_max) / row_sum;
        }
    }
}

// cuda_online_softmax<<<batch, BLOCK_SIZE>>>
__global__ void cuda_online_softmax(const float* __restrict__ x, 
                                    float* __restrict__ y,
                                    const int L) {
    int row = blockIdx.x;
    const int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_num = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    const float* x_row = x + row * L;
    float* y_row = y + row * L;
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        float old_max = local_max;
        local_max = fmaxf(local_max, x_row[i]);
        local_sum = local_sum * expf(old_max - local_max) + expf(x_row[i] - local_max);
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        float other_sum = __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        float new_max = fmaxf(other_max, local_max);
        local_sum = other_sum * expf(other_max - new_max) + local_sum * expf(local_max - new_max);
        local_max = new_max;
    }
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
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
    for (int i = tid; i < L; i += blockDim.x) {
        y_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}