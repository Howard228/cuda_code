#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))
// cuda_transpose_naive<<<grid, block>>>
__global__ void cuda_transpose_naive(const float *input, float *output, const int M, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    output[col * M + row] = input[row * N + col];
}


// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))
// cuda_transpose_naive<32><<<grid, block>>>
template <const int BLOCK_SIZE>
__global__ void cuda_transpose(float *input, float *output, const int M, const int N) {
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    input += by * N + bx;
    output += bx * M + by;
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE + 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (bx + tx < N && by + ty < M) {
        s_mem[ty][tx] = input[ty * N + tx];
    }
    __syncthreads();
    if (ty + bx < N && tx + by < M) {
        output[ty * M + tx] = s_mem[tx][ty];
    }
}

