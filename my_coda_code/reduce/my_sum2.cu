#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>


__device__ float block_sum_reduce(float val) {
    const int warp_size = 32;
    int warp_id = threadIdx.x / warp_size;
    int tid_in_warp = threadIdx.x % warp_size;
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    __shared__ float warp_sums[32];
    if (tid_in_warp == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0) {
        val = (tid_in_warp < blockDim.x / warp_size) ? warp_sums[tid_in_warp] : 0.0f;
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

__global__ void sum_reduce_atomic(const float *x, float *y, const int L, const int block_per_row) {
    int row = blockIdx.x / block_per_row;
    int block_in_row = blockIdx.x % block_per_row;
    int tid = threadIdx.x;
    int tid_in_row = blockDim.x * block_in_row + tid;
    int stride = block_per_row * blockDim.x;
    float sum = 0.0f;
    int L4 = L / 4;
    const float4 *x4 = reinterpret_cast<const float4*>(x + row * L);
    for (int i = tid_in_row; i < L4; i += stride) {
        float4 v = x4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    int end_start = L4 * L;
    if (end_start + tid_in_row< L) {
        sum += x[row * L + end_start + tid_in_row];
    }
    sum = block_sum_reduce(sum);
    if (tid == 0) {
        atomicAdd(&y[row], sum);
    }
}

int main() {
    const int batch = 64;
    const int L = 10000000;
    const int BLOCK_SIZE = 64;
    const int block_per_row = 128;
    const int total_blocks = batch * block_per_row;
    float *d_x, *d_y;
    cudaMalloc(&d_x, batch * L * sizeof(float));
    cudaMalloc(&d_y, batch * sizeof(float));
    sum_reduce_atomic<<<total_blocks, BLOCK_SIZE>>>(d_x, d_y, L, block_per_row);
    return 0;
}