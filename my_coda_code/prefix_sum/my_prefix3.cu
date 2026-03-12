#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>


// cuda_prefix_sum_naive<<<1, L>>>
__global__ void cuda_prefix_sum_naive(const float *input, float *output) {
    float sum = 0.0f;
    for (int i = 0; i <= threadIdx.x; ++i) {
        sum += input[i];
    }
    output[threadIdx.x] = sum;
}



// cuda_prefix_sum<<<1, L / 2, L * sizeof(float)>>>
__global__ void cuda_prefix_sum(const float *input, float *output, const int L) {
    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    if (tid * 2 < L) s_mem[tid * 2] = input[tid * 2];
    if (tid * 2 + 1 < L) s_mem[tid * 2 + 1] = input[tid * 2 + 1];
    int offset = 1;
    for (int valid_thread_num = L >> 1; valid_thread_num > 0; valid_thread_num >>= 1) {
        if (tid < valid_thread_num) {
            int i1 = offset * (tid * 2 + 1) - 1;
            int i2 = offset * (tid * 2 + 2) - 1;
            s_mem[i2] += s_mem[i1];
        }
        offset <<= 1;
        __syncthreads();
    }
    if (tid == 0) s_mem[L - 1] = 0;
    __syncthreads();
    for (int valid_thread_num = 1; valid_thread_num < L; valid_thread_num <<= 1) {
        if (tid < valid_thread_num) {
            int i1 = offset * (tid * 2 + 1) - 1;
            int i2 = offset * (tid * 2 + 2) - 1;
            float tmp = s_mem[i2];
            s_mem[i2] += s_mem[i1];
            s_mem[i1] = tmp;
        }
        offset >>= 1;
        __syncthreads();
    }
    if (tid * 2 < L) output[tid * 2] = input[tid * 2] + s_mem[tid * 2];
    if (tid * 2 + 1 < L) output[tid * 2 + 1] = input[tid * 2 + 1] + s_mem[tid * 2 + 1];
} 