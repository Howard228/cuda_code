#include <cuda_runtime.h>

// cuda_col_sum_reduce<<<CEIL(batch, BLOCK_SIZE)>>>
__global__ void cuda_col_sum_reduce(const float *input, float *output, const int batch, const int L) {
    const int col = threadIdx.x;
    float sum = 0.0f;
    for (int row = blockIdx.x; row < batch; row += gridDim.x) {
        sum += input[row * L + col];
    }
    atomicAdd(&output[col], sum);
}