#define CEIL(a, b) (((a) + (b) - 1) / (b))

template <int BLOCK_SIZE = 32>
__global__ void transpose(const float* input, float* output, int M, int N) {
    
    // ===== 关键：+1 padding 消除 bank conflict =====
    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    // 这个block负责input中哪个区域？
    int bx = blockIdx.x * BLOCK_SIZE;  // 列起点
    int by = blockIdx.y * BLOCK_SIZE;  // 行起点
    
    // Step 1: 读input → 写shared memory
    int col_in = bx + threadIdx.x;  // input的列
    int row_in = by + threadIdx.y;  // input的行
    
    if (col_in < N && row_in < M) {
        // 连续线程读连续地址 → 合并读取 ✅
        smem[threadIdx.y][threadIdx.x] = input[row_in * N + col_in];
    }
    
    __syncthreads();  // 等所有线程写完shared memory
    
    // Step 2: 读shared memory → 写output
    // 转置后：input的[by:by+BS, bx:bx+BS] → output的[bx:bx+BS, by:by+BS]
    int col_out = by + threadIdx.x;  // output的列（注意：by不是bx！）
    int row_out = bx + threadIdx.y;  // output的行
    
    if (col_out < M && row_out < N) {
        // 读shared: smem[threadIdx.x][threadIdx.y]
        //   → padding后，不同线程访问不同bank ✅
        // 写output: 连续线程写连续地址 → 合并写入 ✅
        output[row_out * M + col_out] = smem[threadIdx.x][threadIdx.y];
    }
}

// 调用
// input: M行N列, output: N行M列
// int M = 1024, N = 2048;
// dim3 block(32, 32);
// dim3 grid(CEIL(N, 32), CEIL(M, 32));
// transpose<32><<<grid, block>>>(d_input, d_output, M, N);

// Bank = 物理上独立的存储芯片，每个芯片每周期只能做1次读写
// Shared memory = 32个这样的芯片并排组成
// Bank conflict = 多个线程抢同一个芯片的读写口 → 被迫排队