#include <cuda_runtime.h>

// dim3 grid(CEIL(N, BN), CEIL(M, BM))
// dim3 block((BM * BN) / (TM * TN))
// cuda_sgemm<BM, BN, BK, TM, TN><<<grid, block>>>
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void cuda_sgemm(float *A, float *B, float *C,
                           const int M, const int N, const int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    A += by * BM * K;
    B += bx * BN;
    C += by * BM * N + bx * BN;
    int row_thread_num = BN / TN;
    int total_thread_num = (BM * BN) / (TM * TN);

    int tx = threadIdx.x % row_thread_num * TN;
    int ty = threadIdx.x / row_thread_num * TM;

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = total_thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = total_thread_num / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float accumm[TM][TN] = {0.0f};
    float reg_a[TM], reg_b[TN];

    for (int k_out = 0; k_out < K; k_out += BK) {
        for (int row = 0; row < BM; row += a_tile_stride) {
            As[(a_tile_row + row) * BK + a_tile_col] = A[(a_tile_row + row) * K + a_tile_col];
        }
        for (int row = 0; row < BK; row += b_tile_stride) {
            Bs[(b_tile_row + row) * BN + b_tile_col] = B[(b_tile_row + row) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for(int k_in = 0; k_in < BK; ++k_in) {
            for (int row = 0; row < TM; ++row) {
                reg_a[row] = As[(ty + row) * BK + k_in];
            }
            for (int col = 0; col < TN; ++col) {
                reg_b[col] = Bs[k_in * BN + tx + col];
            }
            for (int row = 0; row < TM; ++row) {
                for (int col = 0; col < TN; ++col) {
                    accumm[row][col] += reg_a[row] * reg_b[col];
                }
            }
        }
        __syncthreads();
    }
    for (int row = 0; row < TM; ++row) {
        for (int col = 0; col < TN; ++col) {
            C[(ty + row) * N + tx + col] = accumm[row][col];
        }
    }
}