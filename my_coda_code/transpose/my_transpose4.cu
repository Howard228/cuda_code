#include <cuda_runtime.h>


// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))
// cuda_transpose_naive<<<grid, block>>>
__global__ void cuda_transpose_naive(const float *input, float *output, const int M, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}


// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))
// cuda_transpose_naive<<<grid, block>>>
template<const int BLOCK_SIZE = 32>
__global__ void cuda_transpose(const float *input, float *output, const int M, const int N) {
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    input += by * N + bx;
    output += bx * M + by;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE + 1];
    if (by + ty < M && bx + tx < N) {
        s_mem[ty][tx] = input[ty * N + tx];
    }
    __syncthreads();
    if (bx + ty < N && by + tx < M) {
        output[ty * M + tx] = s_mem[tx][ty];
    }
}