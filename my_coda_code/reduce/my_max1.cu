#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
// dim3 block(BLOCK_SIZE)
// dim3 grid(batch)
__global__ void reduce_max_naive(const float *x, float *y, const int L) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    x += row * L;
    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);

    float local_max = -INFINITY;
    for (int i = tid; i < L; i += blockDim.x) {
        local_max = fmaxf(x[i], local_max);
    }
    __shared__ float s_mem[32];
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        local_max = fmaxf(__shfl_down_sync(0xFFFFFFFF, local_max, offset), local_max);
    }
    if (lane == 0) s_mem[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        local_max = (lane < warp_num) ? s_mem[lane] : -INFINITY;
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            local_max = fmaxf(__shfl_down_sync(0xFFFFFFFF, local_max, offset), local_max);
        }
        if (lane == 0) y[row] = local_max;
    }
}

__device__ void atomic_max(float *addr, float val) {
    int *addr_int = reinterpret_cast<int*>(addr);
    int old_val = *addr_int;
    int expected;
    do {
        expected = old_val;
        old_val = atomicCAS(addr_int, expected, __float_as_int((fmaxf((__int_as_float(old_val)), val))));
    } while (expected != old_val);
}

__device__ float block_max_reduce(float val) {
    const int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_num = CEIL(blockDim.x, WARP_SIZE);
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(__shfl_down_sync(0xFFFFFFFF, val, offset), val);
    }
    __shared__ float s_mem[32];
    if (lane == 0) s_mem[warp_id] = val;
    __syncthreads(); 
    if (warp_id == 0) {
        val = (lane < warp_num) ? s_mem[lane] : -INFINITY;
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(__shfl_down_sync(0xFFFFFFFF, val, offset), val);
        }
    }
    return val;
}

// dim3 block(BLOCK_SIZE) 
// dim3 grid(blocks_per_row * batch)
// reduce_max<<<grid, block>>>
__global__ void reduce_max(const float *x, float *y, const int L, const int blocks_per_row) {
    int row = blockIdx.x / blocks_per_row;
    int block_id_in_row = blockIdx.x % blocks_per_row;
    int tid = threadIdx.x;
    int tid_in_row = block_id_in_row * blockDim.x + tid;
    int stride = blockDim.x * blocks_per_row;
    x += row * L;
    float local_max = -INFINITY;
    int L4 = L / 4;
    const float4 *x4 = reinterpret_cast<const float4*>(x);
    for (int i = tid_in_row; i < L / 4; i += stride) {
        float4 v = x4[i];
        local_max = fmaxf(local_max, fmaxf(v.x, fmaxf(v.y, fmaxf(v.z, v.w))));
    }
    int end_start = L4 * 4;
    if (end_start + tid_in_row < L) {
        local_max = fmaxf(x[end_start + tid_in_row], local_max);
    }
    local_max = block_max_reduce(local_max);
    if (tid == 0) {
        atomic_max(&y[row], local_max);
    }
}