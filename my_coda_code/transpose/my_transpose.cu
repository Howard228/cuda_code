#include <cuda_runtime.h>

// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))
// cuda_transpose<32><<<grid, block>>>
template<const int BLOCK_SIZE>
__global__ void cuda_transpose(float *input, float *output, const int M, const int N) {
    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE + 1];
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    input += by * N + bx;
    output += bx * M + by;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (by + ty < M && bx + tx < N) {
        smem[ty][tx] = input[ty * N + tx];
    }
    __syncthreads();
    if (bx + ty < N && by + tx < M) {
        output[ty * M + tx] = smem[tx][ty];
    }
}