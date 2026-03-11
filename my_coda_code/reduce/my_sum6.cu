#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
// dim3 block(BLOCK_SIZE)
// dim3 grid(batch)
__global__ void reduce_sum_naive(const float *x, float *y, const int L) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    x += row * L;
    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);

    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        local_sum += x[i];
    }
    __shared__ float s_mem[32];
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0) s_mem[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane < warp_num) ? s_mem[lane] : 0.0f;
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane == 0) y[row] = val;
    }
}

__device__ float block_sum_reduce(float val) {
    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    __shared__ float s_mem[32];
    if (lane == 0) s_mem[warp_id] = val;
    __syncthreads(); 
    if (warp_id == 0) {
        val = (lane < warp_num) ? s_mem[lane] : 0.0f;
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// dim3 block(BLOCK_SIZE) 
// dim3 grid(blocks_per_row * batch)
// reduce_sum<<<grid, block>>>
__global__ void reduce_sum(const float *x, float *y, const int L, const int blocks_per_row) {
    int row = blockIdx.x / blocks_per_row;
    int block_id_in_row = blockIdx.x % blocks_per_row;
    int tid = threadIdx.x;
    int tid_in_row = block_id_in_row * blockDim.x + tid;
    int stride = blockDim.x * blocks_per_row;
    x += row * L;
    float local_sum = 0.0f;
    int L4 = L / 4;
    const float4 *x4 = reinterpret_cast<const float4*>(x);
    for (int i = tid_in_row; i < L / 4; i += stride) {
        float4 v = x4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    int end_start = L4 * 4;
    if (end_start + tid_in_row < L) {
        local_sum += x[end_start + tid_in_row];
    }
    local_sum = block_sum_reduce(local_sum);
    if (tid == 0) {
        atomicAdd(&y[row], local_sum);
    }
}