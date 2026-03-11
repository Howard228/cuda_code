#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

// dim3 block(BLOCK_SIZE, BLOCK_SIZE)
// dim3 grid(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE))
// cuda_sgemm_naive<<<grid, block>>>
__global__ void cuda_sgemm_naive(const float *A, const float *B, float *C, 
                                 const int M, const int N, const int K) {
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    C += by * N + bx;
    A += by * K;
    B += bx;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (bx + tx >= N || by + ty >= M) return;
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[ty * K + i] * B[i * N + tx];
    }
    C[ty * N + tx] = sum;
}



// dim3 block((BM * BN) / (TM * TN))
// dim3 grid(CEIL(N, BN), CEIL(M, BM))
// cuda_sgemm<BM, BN, BK, TM, TN><<<grid, block>>>
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void cuda_sgemm(const float *A, const float *B, float *C, 
                           const int M, const int N, const int K) {
    int by = blockIdx.y * BM;
    int bx = blockIdx.x * BN;
    C += by * N + bx;
    A += by * K;
    B += bx;

    int tid = threadIdx.x;
    int thread_num_row = BN / TN;
    int tx = tid % thread_num_row * TN;
    int ty = tid / thread_num_row * TM;

    int a_tile_row = tid / BK;
    int a_tile_col = tid % BK;
    int a_tile_stride = blockDim.x / BK;

    int b_tile_row = tid / BN;
    int b_tile_col = tid % BN;
    int b_tile_stride = blockDim.x / BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float accumm[TM][TN] = {0.0f};
    float a_reg[TM], b_reg[TN];

    for (int k_out = 0; k_out < K; k_out += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            if (by + a_tile_row + i < M && k_out + a_tile_col < K)
                As[a_tile_row + i][a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
            else
                As[a_tile_row + i][a_tile_col] = 0.0f;
        }
        for (int i = 0; i < BK; i += b_tile_stride) {
            if (k_out + b_tile_row + i < K && bx + b_tile_col < N)
                Bs[b_tile_row + i][b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
            else
                Bs[b_tile_row + i][b_tile_col] = 0.0f;
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k_in = 0; k_in < BK; ++k_in) {
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[ty + i][k_in];
            }
            for (int i = 0; i < TN; ++i) {
                b_reg[i] = Bs[k_in][tx + i];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accumm[i][j] += a_reg[i] * b_reg[j];
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

