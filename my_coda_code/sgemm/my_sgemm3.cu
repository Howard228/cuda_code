#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// dim3 block(32, 32)
// dim3 grid(CEIL(N, 32), CEIL(M, 32))
// cuda_gemm_naive<<<grid, block>>>
__global__ void cuda_gemm_naive(const float *A, const float *B, float *C, 
                                const int M, const int N, const int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


// dim3 grid(CEIL(N, BN), CEIL(M, BM))
// dim3 block((BM * BN) / (TM * TN))
// cuda_gemm<<<grid, block>>>
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void cuda_gemm(const float *A, const float *B, float *C, 
                                const int M, const int N, const int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    A += by * BM * K;
    B += bx * BN;
    C += by * BM * N + bx * BN;
    int tid = threadIdx.x;
    int thread_num_row = BN / TN;
    int total_thread_num = (BM * BN) / (TM * TN);

    int tx = (threadIdx.x % thread_num_row) * TN;
    int ty = (threadIdx.x / thread_num_row) * TM;

    int a_tile_row = tid / BK;
    int a_tile_col = tid % BK;
    int a_tile_stride = total_thread_num / BK;

    int b_tile_row = tid / BN;
    int b_tile_col = tid % BN;
    int b_tile_stride = total_thread_num / BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float accumm[TM][TN] = {0.0f};
    float a_reg[TM], b_reg[TN];

    for (int k_out = 0; k_out < K; k_out += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[a_tile_row + i][a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[b_tile_row + i][b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
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
