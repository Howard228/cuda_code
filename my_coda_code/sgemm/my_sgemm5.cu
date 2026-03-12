#include <cuda_runtime.h>

// dim3 block(BLOCK_SIZE, BLOCK_SIZE)
// dim3 grid(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE))
__global__ void cuda_sgemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}



// dim3 block((BM * BN) / (TM * TN))
// dim3 grid(CEIL(N, BN), CEIL(M, BM))
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void cuda_sgemm(const float *A, const float *B, float *C, int M, int N, int K) {
    int by = blockIdx.y * BM;
    int bx = blockIdx.x * BN;
    int thread_num_total = blockDim.x;
    int thread_num_per_row = BN / TN;
    int ty = threadIdx.x / thread_num_per_row * TM;
    int tx = threadIdx.x % thread_num_per_row * TN;
    A += by * K;
    B += bx;
    C += by * N + bx;

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num_total / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num_total / BN;

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
                a_reg[i] = As[i + ty][k_in];
            }
            for (int i = 0; i < TN; ++i) {
                b_reg[i] = Bs[k_in][i + tx];
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