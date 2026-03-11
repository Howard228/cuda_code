#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>


// dim3 block(BLOCK_SIZE)
// dim3 grid(CEIL(batch, BLOCK_SIZE))
__global__ void reduce_sum_naive(const float *x, float *y, const int L, const int batch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch) return;
    float sum = 0.0f;
    for (int i = 0; i < L; ++i) {
        sum += x[row * L + i];
    }
    y[row] = sum;
}

#define CEIL(a, b) (((a) + (b) - 1) / (b))

__device__ float block_sum_reduce(float val) {
    const int WARP_SIZE = 32;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    __shared__ float s_mem[32];
    if (lane == 0) s_mem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane < warp_num) ? s_mem[lane] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}


// dim3 block(BLOCK_SIZE)
// dim3 grid(blocks_per_row * batch)
__global__ void reduce_sum(const float *x, float *y, const int L, const int blocks_per_row) {
    int row = blockIdx.x / blocks_per_row;
    int block_id_in_row = blockIdx.x % blocks_per_row;
    int tid_in_row = block_id_in_row * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blocks_per_row;
    float local_sum = 0.0f;
    x += row * L;
    const float4 *x4 = reinterpret_cast<const float4*>(x);
    const int L4 = L / 4;
    for (int i = tid_in_row; i < L4; i += stride) {
        float4 v = x4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    int end_start = L4 * 4;
    if (end_start + tid_in_row < L) {
        local_sum += x[end_start + tid_in_row];
    }
    local_sum = block_sum_reduce(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(&y[row], local_sum);
    }

}