#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// ============================================================
// GPU Kernel: Blelloch Scan → 直接输出 inclusive scan
// ============================================================
__global__ void inclusiveScanGPU(int* d_input, int* d_output, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int offset = 1;

    // Step1: 加载到共享内存
    int idx = 2 * tid;
    if (idx < n)     temp[idx]     = d_input[idx];
    if (idx + 1 < n) temp[idx + 1] = d_input[idx + 1];
    __syncthreads();

    // Step2: 上升阶段
    for (int d = n >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
        __syncthreads();
    }

    // Step3: 根置0
    if (tid == 0) temp[n - 1] = 0;
    __syncthreads();

    // Step4: 下降阶段
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // Step5: 写回（直接加上 input，变成 inclusive scan）
    if (idx < n)     d_output[idx]     = temp[idx]     + d_input[idx];
    if (idx + 1 < n) d_output[idx + 1] = temp[idx + 1] + d_input[idx + 1];
    //                                                    ↑ 就改了这里
}

// ============================================================
// GPU Wrapper
// ============================================================
void prefixSumGPU(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();

    // 补成2的幂
    int n_pad = 1;
    while (n_pad < n) n_pad <<= 1;

    std::vector<int> padded_input(n_pad, 0);
    for (int i = 0; i < n; i++) padded_input[i] = input[i];

    int *d_input, *d_output;
    cudaMalloc(&d_input, n_pad * sizeof(int));
    cudaMalloc(&d_output, n_pad * sizeof(int));
    cudaMemcpy(d_input, padded_input.data(), n_pad * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = n_pad / 2;
    int smem_size = n_pad * sizeof(int);
    inclusiveScanGPU<<<1, blockSize, smem_size>>>(d_input, d_output, n_pad);

    // 直接拷回，不需要 CPU 转换了
    output.resize(n);
    cudaMemcpy(output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================
// CPU
// ============================================================
void prefixSumCPU(const std::vector<int>& input, std::vector<int>& output) {
    output.resize(input.size());
    int sum = 0;
    for (size_t i = 0; i < input.size(); i++) {
        sum += input[i];
        output[i] = sum;
    }
}

// ============================================================
// 测试
// ============================================================
int main() {
    {
        std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int> cpu_out, gpu_out;
        prefixSumCPU(input, cpu_out);
        prefixSumGPU(input, gpu_out);

        std::cout << "=== Test 1 (n=8) ===" << std::endl;
        std::cout << "Input:  ";
        for (int v : input) std::cout << v << " ";
        std::cout << std::endl;
        std::cout << "CPU:    ";
        for (int v : cpu_out) std::cout << v << " ";
        std::cout << std::endl;
        std::cout << "GPU:    ";
        for (int v : gpu_out) std::cout << v << " ";
        std::cout << std::endl;
        std::cout << (cpu_out == gpu_out ? "PASSED!" : "FAILED!") << std::endl << std::endl;
    }

    {
        std::vector<int> input = {3, 1, 4, 1, 5};
        std::vector<int> cpu_out, gpu_out;
        prefixSumCPU(input, cpu_out);
        prefixSumGPU(input, gpu_out);

        std::cout << "=== Test 2 (n=5) ===" << std::endl;
        std::cout << "Input:  ";
        for (int v : input) std::cout << v << " ";
        std::cout << std::endl;
        std::cout << "CPU:    ";
        for (int v : cpu_out) std::cout << v << " ";
        std::cout << std::endl;
        std::cout << "GPU:    ";
        for (int v : gpu_out) std::cout << v << " ";
        std::cout << std::endl;
        std::cout << (cpu_out == gpu_out ? "PASSED!" : "FAILED!") << std::endl << std::endl;
    }

    {
        int n = 1024;
        std::vector<int> input(n);
        for (int i = 0; i < n; i++) input[i] = i + 1;
        std::vector<int> cpu_out, gpu_out;
        prefixSumCPU(input, cpu_out);
        prefixSumGPU(input, gpu_out);

        std::cout << "=== Test 3 (n=1024) ===" << std::endl;
        std::cout << (cpu_out == gpu_out ? "PASSED!" : "FAILED!") << std::endl;
    }

    return 0;
}