#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>


__device__ float block_reduce_sum(float val) {
    const int WARP_SIZE = 32;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_num = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane < warp_num) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// sum_reduce_atomic<<<batch * blocks_per_row, BLOCK_SIZE>>>
__global__ void sum_reduce_atomic(const float* __restrict__ x, float* __restrict__ y,
                                  const int L, const int blocks_per_row) {
    int row = blockIdx.x / blocks_per_row;
    int block_id_in_row = blockIdx.x % blocks_per_row;
    int tid = threadIdx.x;
    int tid_in_row = block_id_in_row * blockDim.x + threadIdx.x;
    int stride = blocks_per_row * blockDim.x;
    float local_sum = 0.0f;
    int L4 = L / 4;
    const float4 *x4 = reinterpret_cast<const float4*>(x + row * L);
    for (int i = tid_in_row; i < L4; i += stride) {
        float4 v = x4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    int end_start = L4 * 4;
    if (end_start + tid_in_row < L) {
        local_sum += x[row * L + end_start + tid_in_row];
    }
    local_sum = block_reduce_sum(local_sum);
    if (tid == 0) {
        atomicAdd(&y[row], local_sum);
    }
}