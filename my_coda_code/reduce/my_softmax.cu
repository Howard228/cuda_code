#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <math.h>

void cpu_softmax(const float *x, float *y, int batch, const int L) {
    for (int row = 0; row < batch; row++) {
        float max_val = -INFINITY;
        const float *x_row = x + row * L;
        float *y_row = y + row * L;
        for (int i = 0; i < L; i++) {
            max_val = fmaxf(max_val, x_row[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < L; i++) {
            sum += expf(x_row[i] - max_val);
        }
        for (int i = 0; i < L; i++) {
            y_row[i] = expf(x_row[i] - max_val) / sum;
        } 
    }
}

// cuda_softmax<<<batch, BLOCK_SIZE>>>(d_x, d_y, batch, L)
__global__ void cuda_softmax(const float *x, float *y, const int L) {
    const int WARP_SIZE = 32;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_num = blockDim.x / WARP_SIZE;
    const float *x_row = x + row * L;
    float *y_row = y + row * L;
    float local_max = -INFINITY;
    for (int i = tid; i < L; i += blockDim.x) {
        local_max = fmaxf(local_max, x_row[i]);
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }
    __shared__ float s_max[32];
    if (lane == 0) s_max[warp_id] = local_max;
    __syncthreads();
    float row_max = -INFINITY;
    for (int i = 0; i < warp_num; i++) {
        row_max = max(row_max, s_max[i]);
    }

    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        local_sum += expf(x_row[i] - row_max);
    }
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp_id] = local_sum;
    __syncthreads();
    float row_sum = 0.0f;
    for (int i = 0; i < warp_num; i++) {
        row_sum += s_sum[i];
    }
    for (int i = tid; i < L; i += blockDim.x) {
        y_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}