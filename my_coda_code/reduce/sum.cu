#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                         \
            exit(1);                                                \
        }                                                           \
    } while (0)

// ========== block 内归约 ==========
__inline__ __device__ float block_reduce_sum(float val) {
    const int warpSize = 32;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // warp 内 shuffle 归约， #pragma unroll 把循环展开，不要一圈一圈跳转，直接把每一圈的代码平铺出来
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    // warp 间通过 shared memory
    __shared__ float warpSums[32]; // 32 是一个 block 内最多可能有的 warp 数量。
    if (lane == 0) warpSums[warp_id] = val;
    __syncthreads();

    // warp 0 最终归约
    if (warp_id == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? warpSums[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ========== 单 Kernel：多 block 处理一行 + atomicAdd ==========
__global__ void row_reduce_atomic(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int L,
                                   int blocks_per_row) {
    int row = blockIdx.x / blocks_per_row; // blockIdx.x 就是当前block编号 row 当前线程位于的行
    int block_in_row = blockIdx.x % blocks_per_row; //  blocks_per_row 每行用多少个block并行处理  block_in_row 当前行block的编号
    int tid = threadIdx.x; // 一个block内线程的编号
    int global_tid = block_in_row * blockDim.x + tid; // block_in_row 当前行block的编号  blockDim.x 一个block有多少个线程， global_tid 就是当前线程在当前行的编号
    int stride = blocks_per_row * blockDim.x; // blocks_per_row 每行用多少个block并行处理 blockDim.x 一个block有多少个线程 stride 当前行的所有block数量

    float sum = 0.0f;

    // float4 向量化加载
    int L4 = L / 4; // L 一行的总元素数量
    const float4* x4 = reinterpret_cast<const float4*>(x + row * L); // row 当前线程位于的行  row * L 当前行前面有多少个元素 x4 定位到当前行的首元素
    for (int i = global_tid; i < L4; i += stride) { // global_tid 就是当前线程在当前行的编号
        float4 v = x4[i];
        sum += v.x + v.y + v.z + v.w;
    }

    // 尾部处理
    int tail_start = L4 * 4;
    if (tail_start + global_tid < L)
        sum += x[row * L + tail_start + global_tid];

    // block 内归约
    sum = block_reduce_sum(sum);

    // 原子累加到最终结果
    if (tid == 0) {
        atomicAdd(&y[row], sum);
    }
}

// ========== CPU 参考实现 ==========
void cpu_row_reduce(const float* x, float* y, int batch, int L) {
    for (int row = 0; row < batch; row++) {
        float sum = 0.0f;
        for (int i = 0; i < L; i++) {
            sum += x[row * L + i];
        }
        y[row] = sum;
    }
}

// ========== 主函数 ==========
int main() {
    // -------- 参数 --------
    const int batch = 64;
    const int L = 10000000;        // 每行 1000万
    const int BLOCK_SIZE = 256;
    const int blocks_per_row = 128;
    const int total_blocks = batch * blocks_per_row;  // 64 * 128 = 8192

    printf("=== Row Reduce (atomicAdd, single kernel) ===\n");
    printf("batch           = %d\n", batch);
    printf("L               = %d (%.1fM per row)\n", L, L / 1e6);
    printf("BLOCK_SIZE      = %d\n", BLOCK_SIZE);
    printf("blocks_per_row  = %d\n", blocks_per_row);
    printf("total_blocks    = %d\n", total_blocks);
    printf("elem per thread ≈ %d\n\n", L / (blocks_per_row * BLOCK_SIZE));

    // -------- 主机内存 --------
    size_t bytes_x = (size_t)batch * L * sizeof(float);
    size_t bytes_y = batch * sizeof(float);

    float* h_x     = (float*)malloc(bytes_x);
    float* h_y_gpu = (float*)malloc(bytes_y);
    float* h_y_cpu = (float*)malloc(bytes_y);

    if (!h_x || !h_y_gpu || !h_y_cpu) {
        printf("Host malloc failed!\n");
        return 1;
    }

    // -------- 初始化数据 --------
    srand(42);
    for (long long i = 0; i < (long long)batch * L; i++) {
        h_x[i] = (float)(rand() % 100) / 100.0f;  // [0, 1)
    }

    // -------- CPU 参考 --------
    printf("Running CPU reference... ");
    cpu_row_reduce(h_x, h_y_cpu, batch, L);
    printf("done\n");
    printf("CPU first 5: ");
    for (int i = 0; i < 5 && i < batch; i++)
        printf("%.2f ", h_y_cpu[i]);
    printf("\n\n");

    // -------- 设备内存 --------
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, bytes_x));
    CHECK_CUDA(cudaMalloc(&d_y, bytes_y));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));

    // -------- 预热 --------
    CHECK_CUDA(cudaMemset(d_y, 0, bytes_y));  // ★ 清零 ★
    row_reduce_atomic<<<total_blocks, BLOCK_SIZE>>>(d_x, d_y, L, blocks_per_row);
    CHECK_CUDA(cudaDeviceSynchronize());

    // -------- 正式计时 --------
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int nIter = 100;
    CHECK_CUDA(cudaEventRecord(start));

    for (int iter = 0; iter < nIter; iter++) {
        // ★★★ 每次迭代都要清零 y ★★★
        CHECK_CUDA(cudaMemset(d_y, 0, bytes_y));

        // ★★★ 只需一次 kernel launch ★★★
        row_reduce_atomic<<<total_blocks, BLOCK_SIZE>>>(d_x, d_y, L, blocks_per_row);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float msecTotal = 0;
    CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));
    float msecPerIter = msecTotal / nIter;

    // -------- 拷贝结果 --------
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, bytes_y, cudaMemcpyDeviceToHost));

    printf("GPU first 5: ");
    for (int i = 0; i < 5 && i < batch; i++)
        printf("%.2f ", h_y_gpu[i]);
    printf("\n\n");

    // -------- 验证 --------
    bool correct = true;
    float max_rel_err = 0.0f;
    for (int i = 0; i < batch; i++) {
        float rel_err = fabs(h_y_gpu[i] - h_y_cpu[i]) / fabs(h_y_cpu[i]);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err > 1e-3) {
            printf("  MISMATCH row %d: GPU=%.6f CPU=%.6f rel_err=%.2e\n",
                   i, h_y_gpu[i], h_y_cpu[i], rel_err);
            correct = false;
        }
    }
    printf("Max relative error: %.2e\n", max_rel_err);
    printf("Result: %s\n\n", correct ? "PASS ✅" : "FAIL ❌");

    // -------- 性能 --------
    double totalBytes = (double)batch * L * sizeof(float);
    double bandwidth = totalBytes / (msecPerIter / 1000.0) / 1e9;
    printf("Performance:\n");
    printf("  Time:      %.3f ms/iter\n", msecPerIter);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);

    // -------- 清理 --------
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    free(h_x);
    free(h_y_gpu);
    free(h_y_cpu);

    return 0;
}
