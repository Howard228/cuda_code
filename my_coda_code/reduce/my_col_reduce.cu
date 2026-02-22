#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

// col_sum_reduce<<<256, L>>>
__global__ void col_sum_reduce(const float* __restrict__ x, float* __restrict__ y,
                               const int batch, int L) {
    int col = threadIdx.x;
    if (col >= L) return;
    float sum = 0.0f;
    for (int row = blockIdx.x; row < batch; row += gridDim.x) {
        sum += x[row * L + col];
    }
    atomicAdd(&y[col], sum);
}