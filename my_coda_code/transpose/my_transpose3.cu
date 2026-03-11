#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))
__global__ void cuda_transpose_naive(const float* x, float *y, const int M, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        y[col * M + row] = x[row * N + col];
    }
}




// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))  
// x [M, N] N[N, M]
template <const int BLOCK_SIZE = 32>
__global__ void cuda_transpose(const float* x, float *y, const int M, const int N) {
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    x += by * N + bx;
    y += bx * M + by;
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE + 1];
    if (by + ty < M && bx + tx < N) {
        s_mem[ty][tx] = x[ty * N + tx];
    }
    __syncthreads();
    if (by + tx < M && bx + ty < N) {
        y[ty * M + tx] = s_mem[tx][ty];
    }
}