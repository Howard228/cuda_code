#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>




// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))
// cuda_transpose_naive<<grid, block>>>
__global__ void cuda_transpose_naive(const float *input, float *output, const int M, const int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    output[col * M + row] = input[row * N + col];
}




// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))
// cuda_transpose<<grid, block>>>
template <const int BLOCK_SIZE = 32>
__global__ void cuda_transpose(const float *input, float *output, const int M, const int N) {
    int by = blockIdx.y * blockDim.y;
    int bx = blockIdx.x * blockDim.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    __shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE + 1];
    if (by + ty < M && bx + tx < N) {
        s_mem[ty][tx] = input[(by + ty) * N + bx + tx];
    }
    __syncthreads();
    if (by + tx < M && bx + ty < N) {
        output[(bx + ty) * M + by + tx] = s_mem[tx][ty];
    }
}