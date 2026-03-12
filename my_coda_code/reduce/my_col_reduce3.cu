#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>


// cuda_col_sum_reduce<<<CEIL(batch, L), L>>>
__global__ void cuda_col_sum_reduce(const float *x, float *y, const int batch) {
    int col = threadIdx.x;
    float sum = 0.0f;
    for (int row = blockIdx.x; row < batch; row += gridDim.x) {
        sum += x[row * blockDim.x + col]; 
    }
    atomicAdd(&y[col], sum);
}