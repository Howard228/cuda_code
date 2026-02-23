#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

// dim3 grid(CEIL(N, BN), CEIL(M, BM))
// dim3 block((BM * BN) / (TM * TN))
// cuda_sgemm<BM, BN, BK, TM, TN><<<grid, block>>>
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void cuda_sgemm(const float *A, const float *B, float *C, 
                           const int M, const int N, const int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    A += by * BM * K;
    B += bx * BN;
    C += by * BM * N + bx * BN;
    int tid = threadIdx.x;
    int row_thread_num = BN / TN;
    int total_thread_num = (BM * BN) / (TM * TN);

    int tx = (tid % row_thread_num) * TN;
    int ty = (tid / row_thread_num) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = total_thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = total_thread_num / BN;

    float accumm[TM][TN] = {0.0f};
    float reg_a[TM], reg_b[TN];

    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k_in = 0; k_in < BK; ++k_in) {
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[(ty + i) * BK + k_in];
            }
            for (int i = 0; i < TN; ++i) {
                reg_b[i] = Bs[k_in * BN + i + tx];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accumm[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            C[(ty + i) * N + tx + j] = accumm[i][j];
        }
    } 
    
}