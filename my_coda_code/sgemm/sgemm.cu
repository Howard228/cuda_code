// 面试时先写出这些参数，告诉面试官设计意图
// BM=128, BN=128: block tile 大小
// BK=8: K方向分块大小
// TM=8, TN=8: 每线程计算 8×8 = 64 个元素
// 线程数 = 128×128/(8×8) = 256

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 每线程在 block tile 中的位置
    int block_row_thread = BN / TN;  // 一行多少线程
    int thread_num = (BM / TM) * (BN / TN);  // block 总线程数

    int tx = (threadIdx.x % block_row_thread) * TN;  // 列偏移
    int ty = (threadIdx.x / block_row_thread) * TM;  // 行偏移

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动指针到当前 block tile 起始位置
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // 搬运数据用的索引
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    // 寄存器
    float accum[TM][TN] = {0.0f};
    float reg_a[TM], reg_b[TN];

    // K 方向分块循环
    for (int k = 0; k < K; k += BK) {
        // global → shared（多次搬运覆盖整个 tile）
        for (int i = 0; i < BM; i += a_tile_stride)
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        for (int i = 0; i < BK; i += b_tile_stride)
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        __syncthreads();

        A += BK;
        B += BK * N;

        // 计算：BK 在外层，寄存器缓存 + 外积
        for (int i = 0; i < BK; i++) {
            for (int r = 0; r < TM; r++)
                reg_a[r] = As[(ty + r) * BK + i];
            for (int c = 0; c < TN; c++)
                reg_b[c] = Bs[i * BN + (tx + c)];
            for (int row = 0; row < TM; row++)
                for (int col = 0; col < TN; col++)
                    accum[row][col] += reg_a[row] * reg_b[col];
        }
        __syncthreads();
    }

    // 写回 C
    for (int row = 0; row < TM; row++)
        for (int col = 0; col < TN; col++)
            C[(ty + row) * N + (tx + col)] = accum[row][col];
}

// 调用
// dim3 block(256);
// dim3 grid(CEIL(N,128), CEIL(M,128));
// sgemm<128, 128, 8, 8, 8><<<grid, block>>>(A, B, C, M, N, K);