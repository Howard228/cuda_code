#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>


// cuda_prefix_sum_naive<<<1, L>>>
__global__ void cuda_prefix_sum_naive(const float *x, float *y, const int L) {
    const int col = threadIdx.x;
    if (col >= L) return;
    float sum = 0.0f;
    for (int i = 0; i <= col; ++i) {
        sum += x[i];
    }
    y[col] = sum;
}



// cuda_prefix_sum_naive<<<1, L / 2, L * sizeof(float)>>>
__global__ void cuda_prefix_sum(const float *x, float *y, const int L) {
    extern __shared__ float s_mem[];
    int tid = threadIdx.x;
    int idx1 = tid * 2;
    int idx2 = tid * 2 + 1;
    if (idx1 < L) s_mem[idx1] = x[idx1];
    if (idx2 < L) s_mem[idx2] = x[idx2];
    __syncthreads();
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
    if (tid == 0) s_mem[L - 1] = 0.0f;
    __syncthreads();
    for (int valid_thread_num = 1; valid_thread_num < L; valid_thread_num <<= 1) {
        offset >>= 1;
        if (tid < valid_thread_num) {
            int i1 = offset * (tid * 2 + 1) - 1;
            int i2 = offset * (tid * 2 + 2) - 1;
            float tmp = s_mem[i2];
            s_mem[i2] = tmp + s_mem[i1];
            s_mem[i1] = tmp;
        }
        __syncthreads();
    }
    if (idx1 < L) y[idx1] = x[idx1] + s_mem[idx1];
    if (idx2 < L) y[idx2] = x[idx2] + s_mem[idx2];
}