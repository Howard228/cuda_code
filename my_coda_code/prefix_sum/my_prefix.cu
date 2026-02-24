#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>


// cuda_prefix_sum_naive<<<1, L>>>(input, output, L)
// 每个线程负责一个位置，直接暴力累加前面所有元素
__global__ void cuda_prefix_sum_naive(const float *input, float *output, const int L) {
    int tid = threadIdx.x;
    if (tid < L) {
        float sum = 0.0f;
        for (int i = 0; i <= tid; i++) {  // 从头加到自己
            sum += input[i];
        }
        output[tid] = sum;
    }
}

// cuda_prefix_sum<<<1, L / 2, L * sizeof(float)>>> 注意：L是pad后2的幂
__global__ void cuda_prefix_sum(const float *input, float *output, const int L) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int idx1 = threadIdx.x * 2;
    int idx2 = threadIdx.x * 2 + 1;
    if (idx1 < L) temp[idx1] = input[idx1];
    if (idx2 < L) temp[idx2] = input[idx2];
    __syncthreads();
    int offset = 1;
    for (int valid_thread_nums = L >> 1; valid_thread_nums > 0; valid_thread_nums >>= 1) {
        if (tid < valid_thread_nums) {
            int i1 = offset * (tid * 2 + 1) - 1;
            int i2 = offset * (tid * 2 + 2) - 1;
            temp[i2] += temp[i1];
        }
        offset <<= 1;
        __syncthreads();
    }
    if (tid == 0) temp[L - 1] = 0.0f;
    __syncthreads();
    for (int valid_thread_nums = 1; valid_thread_nums < L; valid_thread_nums <<= 1) {
        if (tid < valid_thread_nums) {
            int i1 = offset * (tid * 2 + 1) - 1;
            int i2 = offset * (tid * 2 + 2) - 1;
            float tmp = temp[i2];
            temp[i2] += temp[i1];
            temp[i1] = tmp;
        }
        offset >>= 1;
        __syncthreads();
    }
    if (idx1 < L) output[idx1] = input[idx1] + temp[idx1];
    if (idx2 < L) output[idx2] = input[idx2] + temp[idx2];
}