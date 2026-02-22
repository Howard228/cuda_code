#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>


__device__ float block_reduce_sum(float val) {
    const int warpSize = 32;
    int tid_in_warp = threadIdx.x % 32;
    int warp_id = threadIdx.x / warpSize;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    __shared__ float warpSums[32];
    if (tid_in_warp == 0) warpSums[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? warpSums[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}



__global__ void row_reduce_atomic(const float *x, 
                                  float *y, 
                                  const int L, 
                                  const int block_per_row) {
    int row = blockIdx.x / block_per_row;
    int block_in_row = blockIdx.x % block_per_row;
    int tid = threadIdx.x;
    int tid_in_row = block_in_row * blockDim.x + tid;
    int stride = block_per_row * blockDim.x;
    float sum = 0.0f;

    int L4 = L / 4;
    const float4 *x4 = reinterpret_cast<const float4*>(x + row * L);
    for (int i = tid_in_row; i < L4; i += stride) {
        float4 v = x4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    int end_start = L4 * 4;
    if (end_start + tid_in_row < L) {
        sum += x[row * L + end_start + tid_in_row];
    }

    sum = block_reduce_sum(sum);

    if (tid == 0) {
        atomicAdd(&y[row], sum);
    }

}




int main() {
    const int batch = 64;
    const int L = 1000000;
    const int BLOCK_SIZE = 256;
    const int blocks_per_row = 128;
    const int total_blocks = batch * blocks_per_row;
    float *d_x, *d_y;
    cudaMalloc(&d_x, batch * L * sizeof(float));
    cudaMalloc(&d_y, batch * sizeof(float));
    row_reduce_atomic<<<total_blocks, BLOCK_SIZE>>>(d_x, d_y, L, blocks_per_row);
    return 0;
}


