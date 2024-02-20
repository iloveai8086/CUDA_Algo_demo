#include "test.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cub/cub.cuh>
#include "../tensorRT/common/ilogger.hpp"
#include "../tensorRT/common/cuda_tools.hpp"
#include "../tools/timer.hpp"

// ---------------------------------------------------------------------------
__global__ void ScanAndWritePartSumKernel(const int32_t* input, int32_t* part,
                                          int32_t* output, size_t n, size_t part_num) {
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        size_t part_begin = part_i * blockDim.x;
        size_t part_end = min((part_i + 1) * blockDim.x, n);
        if (threadIdx.x == 0) {
            int32_t acc = 0;
            for (size_t i = part_begin; i < part_end; ++i) {
                acc += input[i];
                output[i] = acc;
            }
            part[part_i] = acc;
        }
    }
}

__global__ void ScanPartSumKernel(int32_t* part, size_t part_num) {
    int32_t acc = 0;
    for (size_t i = 0; i < part_num; ++i) {
        acc += part[i];
        part[i] = acc;
    }
}

__global__ void AddBaseSumKernel(int32_t* part, int32_t* output, size_t n,
                                 size_t part_num) {
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        if (part_i == 0) {
            continue;
        }
        int32_t index = part_i * blockDim.x + threadIdx.x;
        if (index < n) {
            output[index] += part[part_i - 1];
        }
    }
}

void ScanThenFan(const int32_t* input, int32_t* buffer, int32_t* output,
                 size_t n) {
    size_t part_size = 1024;
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    // use buffer[0:part_num] to save the metric of part
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    printf("n = %zu, part_size = %zu, part_num = %zu, block_num = %zu\n", n,
           part_size, part_num, block_num);
    checkCudaKernel(ScanAndWritePartSumKernel<<<block_num, part_size>>>(input, part, output,
                                                                        n, part_num));
    // after following step, part[i] = part_sum[0] + part_sum[1] + ... part_sum[i]
    checkCudaKernel(ScanPartSumKernel<<<1, 1>>>(part, part_num));
    // make final result
    checkCudaKernel(AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num));
}

void PrefixSum(const int32_t* input, size_t n, int32_t* output) {
    int32_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += input[i];
        output[i] = sum;
    }
}

// ------------------------------------------------------------------------------
__device__ void ScanBlockV1(int32_t* shm) {
    if (threadIdx.x == 0) {
        int32_t acc = 0;
        for (size_t i = 0; i < blockDim.x; ++i) {
            acc += shm[i];
            shm[i] = acc;
        }
    }
    __syncthreads();
}

__global__ void ScanAndWritePartSumKernelV2(const int32_t* input, int32_t* part,
                                            int32_t* output, size_t n,
                                            size_t part_num) {
    extern __shared__ int32_t shm[];
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        // store this part input to shm
        size_t index = part_i * blockDim.x + threadIdx.x;
        shm[threadIdx.x] = index < n ? input[index] : 0;
        __syncthreads();
        // scan on shared memory
        ScanBlockV1(shm);
        __syncthreads();
        // write result
        if (index < n) {
            output[index] = shm[threadIdx.x];
        }
        if (threadIdx.x == blockDim.x - 1) {
            part[part_i] = shm[threadIdx.x];
        }
    }
}

void ScanThenFanV2(const int32_t* input, int32_t* buffer, int32_t* output,
                   size_t n) {
    size_t part_size = 1024;
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    // use buffer[0:part_num] to save the metric of part,说明buffer的大小就是part_num
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    printf("n = %zu, part_size = %zu, part_num = %zu, block_num = %zu\n", n,
           part_size, part_num, block_num);
    checkCudaKernel(ScanAndWritePartSumKernelV2<<<block_num, part_size, part_size * sizeof(int32_t), nullptr>>>
                            (input, part, output, n, part_num));
    // after following step, part[i] = part_sum[0] + part_sum[1] + ... part_sum[i]
    checkCudaKernel(ScanPartSumKernel<<<1, 1>>>(part, part_num));
    // make final result
    checkCudaKernel(AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num));
}

// ------------------------------------------------------------------------------
__device__ void ScanWarpV1(int32_t* shm_data, int32_t lane) {
    if (lane == 0) {
        int32_t acc = 0;
        for (int32_t i = 0; i < 32; ++i) {
            acc += shm_data[i];
            shm_data[i] = acc;
        }
    }
    __syncwarp();
}

__device__ void ScanWarpV2(int32_t* shm_data) {
    int32_t lane = threadIdx.x & 31;
    volatile int32_t* vshm_data = shm_data;
    if (lane >= 1) vshm_data[0] += vshm_data[-1];
    __syncwarp();
    if (lane >= 2) vshm_data[0] += vshm_data[-2];
    __syncwarp();
    if (lane >= 4) vshm_data[0] += vshm_data[-4];
    __syncwarp();
    if (lane >= 8) vshm_data[0] += vshm_data[-8];
    __syncwarp();
    if (lane >= 16) vshm_data[0] += vshm_data[-16];
    __syncwarp();
}

__device__ void ScanBlockV2(int32_t* shm_data) {
    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane = threadIdx.x & 31;
    __shared__ int32_t warp_sum[32];
    ScanWarpV1(shm_data + 32 * warp_id, lane);
    // ScanWarpV2(shm_data);
    __syncthreads();
    // write sum of each warp to warp_sum
    if (lane == 31) {
        warp_sum[warp_id] = *(shm_data + 32 * warp_id + 31);
    }
    __syncthreads();
    // use a single warp to scan warp_sum
    if (warp_id == 0) {
        ScanWarpV1(warp_sum, lane);
    }
    __syncthreads();
    // add base
    if (warp_id > 0) {
        // *shm_data += warp_sum[warp_id - 1];
        shm_data[threadIdx.x] += warp_sum[warp_id - 1];
    }
    __syncthreads();
}

__global__ void ScanAndWritePartSumKernelV3(const int32_t* input, int32_t* part,
                                            int32_t* output, size_t n,
                                            size_t part_num) {
    extern __shared__ int32_t shm[];
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        // store this part input to shm
        size_t index = part_i * blockDim.x + threadIdx.x;
        shm[threadIdx.x] = index < n ? input[index] : 0;
        __syncthreads();
        // scan on shared memory
        ScanBlockV2(shm);
        __syncthreads();
        // write result
        if (index < n) {
            output[index] = shm[threadIdx.x];
        }
        if (threadIdx.x == blockDim.x - 1) {
            part[part_i] = shm[threadIdx.x];
        }
    }
}

void ScanThenFanV3(const int32_t* input, int32_t* buffer, int32_t* output,
                   size_t n) {
    size_t part_size = 1024;  // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    // use buffer[0:part_num] to save the metric of part
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    printf("n = %zu, part_size = %zu, part_num = %zu, block_num = %zu\n", n,
           part_size, part_num, block_num);
    checkCudaKernel(ScanAndWritePartSumKernelV3<<<block_num, part_size, (part_size) * sizeof(int32_t), nullptr>>>
                            (input, part, output, n, part_num));
    // after following step, part[i] = part_sum[0] + part_sum[1] + ... part_sum[i]
    checkCudaKernel(ScanPartSumKernel<<<1, 1>>>(part, part_num));
    // make final result
    checkCudaKernel(AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num));
}

// ------------------------------------------------------------------------------
__device__ void ScanBlockV3(int32_t* shm_data) {
    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane = threadIdx.x & 31;  // 31 = 00011111
    __shared__ int32_t warp_sum[32];  // blockDim.x / WarpSize = 32 -> 1024 / 32 = 32
    // scan each warp
    ScanWarpV2(shm_data + lane + 32 * warp_id);
    // ScanWarpV2(shm_data);
    __syncthreads();
    // write sum of each warp to warp_sum
    if (lane == 31) {
        // warp_sum[warp_id] = *(shm_data + 31);
        warp_sum[warp_id] = *(shm_data + lane + 32 * warp_id);
    }
    __syncthreads();
    // use a single warp to scan warp_sum
    if (warp_id == 0) {
        ScanWarpV2(warp_sum + lane);
    }
    __syncthreads();
    // add base
    if (warp_id > 0) {
        // *shm_data += warp_sum[warp_id - 1];
        shm_data[threadIdx.x] += warp_sum[warp_id - 1];
    }
    __syncthreads();
}

__global__ void ScanAndWritePartSumKernelV4(const int32_t* input, int32_t* part,
                                            int32_t* output, size_t n,
                                            size_t part_num) {
    extern __shared__ int32_t shm[];
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        // store this part input to shm
        size_t index = part_i * blockDim.x + threadIdx.x;
        shm[threadIdx.x] = index < n ? input[index] : 0;
        __syncthreads();
        // scan on shared memory
        ScanBlockV3(shm);
        __syncthreads();
        // write result
        if (index < n) {
            output[index] = shm[threadIdx.x];
        }
        if (threadIdx.x == blockDim.x - 1) {
            part[part_i] = shm[threadIdx.x];
        }
    }
}

void ScanThenFanV4(const int32_t* input, int32_t* buffer, int32_t* output,
                   size_t n) {
    size_t part_size = 1024;  // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    // use buffer[0:part_num] to save the metric of part
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    printf("n = %zu, part_size = %zu, part_num = %zu, block_num = %zu\n", n,
           part_size, part_num, block_num);
    checkCudaKernel(ScanAndWritePartSumKernelV4<<<block_num, part_size, (part_size) * sizeof(int32_t), nullptr>>>
                            (input, part, output, n, part_num));
    // after following step, part[i] = part_sum[0] + part_sum[1] + ... part_sum[i]
    checkCudaKernel(ScanPartSumKernel<<<1, 1>>>(part, part_num));
    // make final result
    checkCudaKernel(AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num));
}

// ------------------------------------------------------------------------------
__device__ void ScanBlockV4(int32_t* shm_data) {
    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane = threadIdx.x & 31;       // 31 = 00011111
    extern __shared__ int32_t warp_sum[];  // warp_sum[32]
    // scan each warp
    ScanWarpV2(shm_data);
    __syncthreads();
    // write sum of each warp to warp_sum
    if (lane == 31) {
        warp_sum[warp_id] = *shm_data;
    }
    __syncthreads();
    // use a single warp to scan warp_sum
    if (warp_id == 0) {
        ScanWarpV2(warp_sum + lane);
    }
    __syncthreads();
    // add base
    if (warp_id > 0) {
        *shm_data += warp_sum[warp_id - 1];
    }
    __syncthreads();
}

__global__ void ScanAndWritePartSumKernelV5(const int32_t* input, int32_t* part,
                                            int32_t* output, size_t n,
                                            size_t part_num) {
    // the first 32 is used to save warp sum
    extern __shared__ int32_t shm[];
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        // store this part input to shm
        size_t index = part_i * blockDim.x + threadIdx.x;
        shm[32 + threadIdx.x] = index < n ? input[index] : 0;
        __syncthreads();
        // scan on shared memory
        ScanBlockV4(shm + 32 + threadIdx.x);
        __syncthreads();
        // write result
        if (index < n) {
            output[index] = shm[32 + threadIdx.x];
        }
        if (threadIdx.x == blockDim.x - 1) {
            part[part_i] = shm[32 + threadIdx.x];
        }
    }
}

void ScanThenFanV5(const int32_t* input, int32_t* buffer,
                   int32_t* output, size_t n) {
    size_t part_size = 1024;  // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    // use buffer[0:part_num] to save the metric of part
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    size_t shm_size = (32 + part_size) * sizeof(int32_t);
    ScanAndWritePartSumKernelV5<<<block_num, part_size, shm_size>>>(
            input, part, output, n, part_num);
    // after following step, part[i] = part_sum[0] + part_sum[1] + ... part_sum[i]
    ScanPartSumKernel<<<1, 1>>>(part, part_num);
    // make final result
    AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num);
}

// ------------------------------------------------------------------------------
__device__ void ScanWarpV3(int32_t* shm_data) {
    volatile int32_t* vshm_data = shm_data;
    vshm_data[0] += vshm_data[-1];
    vshm_data[0] += vshm_data[-2];
    vshm_data[0] += vshm_data[-4];
    vshm_data[0] += vshm_data[-8];
    vshm_data[0] += vshm_data[-16];
}

__device__ void ScanBlockV5(int32_t* shm_data) {
    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane = threadIdx.x & 31;
    extern __shared__ int32_t warp_sum[];  // 16 zero padding
    // scan each warp
    ScanWarpV3(shm_data);
    __syncthreads();
    // write sum of each warp to warp_sum
    if (lane == 31) {
        warp_sum[16 + warp_id] = *shm_data;
    }
    __syncthreads();
    // use a single warp to scan warp_sum
    if (warp_id == 0) {
        ScanWarpV3(warp_sum + 16 + lane);
    }
    __syncthreads();
    // add base
    if (warp_id > 0) {
        *shm_data += warp_sum[16 + warp_id - 1];
    }
    __syncthreads();
}

__global__ void ScanAndWritePartSumKernelV6(const int32_t* input, int32_t* part,
                                            int32_t* output, size_t n, size_t part_num) {
    // the first 16 + 32 is used to save warp sum
    extern __shared__ int32_t shm[];
    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane = threadIdx.x & 31;
    // initialize the zero padding
    if (threadIdx.x < 16) {
        shm[threadIdx.x] = 0;
    }
    if (lane < 16) {
        shm[(16 + 32) + warp_id * (16 + 32) + lane] = 0;
    }
    __syncthreads();
    // process each part
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        // store this part input to shm
        size_t index = part_i * blockDim.x + threadIdx.x;
        int32_t* myshm = shm + (16 + 32) + warp_id * (16 + 32) + 16 + lane;
        *myshm = index < n ? input[index] : 0;
        __syncthreads();
        // scan on shared memory
        ScanBlockV5(myshm);
        __syncthreads();
        // write result
        if (index < n) {
            output[index] = *myshm;
        }
        if (threadIdx.x == blockDim.x - 1) {
            part[part_i] = *myshm;
        }
    }
}

void ScanThenFanV6(const int32_t* input, int32_t* buffer,
                   int32_t* output, size_t n) {
    size_t part_size = 1024;  // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    // use buffer[0:part_num] to save the metric of part
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    size_t warp_num = part_size / 32;
    size_t shm_size = (16 + 32 + warp_num * (16 + 32)) * sizeof(int32_t);
    ScanAndWritePartSumKernelV6<<<block_num, part_size, shm_size>>>(
            input, part, output, n, part_num);
    // after following step, part[i] = part_sum[0] + part_sum[1] + ... part_sum[i]
    ScanPartSumKernel<<<1, 1>>>(part, part_num);
    // make final result
    AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num);
}

// ------------------------------------------------------------------------------
void ScanThenFanV7(const int32_t* input, int32_t* buffer,
                   int32_t* output, size_t n) {
    size_t part_size = 1024;  // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    // use buffer[0:part_num] to save the metric of part
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    size_t warp_num = part_size / 32;
    size_t shm_size = (16 + 32 + warp_num * (16 + 32)) * sizeof(int32_t);
    ScanAndWritePartSumKernelV6<<<block_num, part_size, shm_size>>>(
            input, part, output, n, part_num);
    if (part_num >= 2) {
        // after following step
        // part[i] = part_sum[0] + part_sum[1] + ... + part_sum[i]
        ScanThenFanV7(part, buffer + part_num, part, part_num);
        // make final result
        AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num);
    }
}

// ------------------------------------------------------------------------------
__device__ int32_t ScanWarpV4(int32_t val) {
    int32_t lane = threadIdx.x & 31;
    int32_t tmp = __shfl_up_sync(0xffffffff, val, 1);
    if (lane >= 1) {
        val += tmp;
    }
    tmp = __shfl_up_sync(0xffffffff, val, 2);
    if (lane >= 2) {
        val += tmp;
    }
    tmp = __shfl_up_sync(0xffffffff, val, 4);
    if (lane >= 4) {
        val += tmp;
    }
    tmp = __shfl_up_sync(0xffffffff, val, 8);
    if (lane >= 8) {
        val += tmp;
    }
    tmp = __shfl_up_sync(0xffffffff, val, 16);
    if (lane >= 16) {
        val += tmp;
    }
    return val;
}

__device__ __forceinline__ int32_t ScanWarpV5(int32_t val) {
    int32_t result;
    asm("{"
        ".reg .s32 r<5>;"
        ".reg .pred p<5>;"

        "shfl.sync.up.b32 r0|p0, %1, 1, 0, -1;"
        "@p0 add.s32 r0, r0, %1;"

        "shfl.sync.up.b32 r1|p1, r0, 2, 0, -1;"
        "@p1 add.s32 r1, r1, r0;"

        "shfl.sync.up.b32 r2|p2, r1, 4, 0, -1;"
        "@p2 add.s32 r2, r2, r1;"

        "shfl.sync.up.b32 r3|p3, r2, 8, 0, -1;"
        "@p3 add.s32 r3, r3, r2;"

        "shfl.sync.up.b32 r4|p4, r3, 16, 0, -1;"
        "@p4 add.s32 r4, r4, r3;"

        "mov.s32 %0, r4;"
        "}"
            : "=r"(result)
            : "r"(val));
    return result;
}

__device__ void ScanBlockV6(int32_t* shm_data) {
    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane = threadIdx.x & 31;       // 31 = 00011111
    __shared__ int32_t warp_sum[32];  // warp_sum[32]
    // scan each warp
    *shm_data = ScanWarpV5(*shm_data);
    __syncthreads();
    // write sum of each warp to warp_sum
    if (lane == 31) {
        warp_sum[warp_id] = *(shm_data);
    }
    __syncthreads();
    // use a single warp to scan warp_sum
    if (warp_id == 0) {
        *(warp_sum + lane) = ScanWarpV5(*(warp_sum + lane));
    }
    __syncthreads();
    // add base
    if (warp_id > 0) {
        *shm_data += warp_sum[warp_id - 1];
    }
    __syncthreads();
}

__device__ __forceinline__ int32_t ScanBlockV7(int32_t val) {
    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane = threadIdx.x & 31;
    extern __shared__ int32_t warp_sum[];
    // scan each warp
    val = ScanWarpV5(val);
    __syncthreads();
    // write sum of each warp to warp_sum
    if (lane == 31) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();
    // use a single warp to scan warp_sum
    if (warp_id == 0) {
        warp_sum[lane] = ScanWarpV5(warp_sum[lane]);
    }
    __syncthreads();
    // add base
    if (warp_id > 0) {
        val += warp_sum[warp_id - 1];
    }
    __syncthreads();
    return val;
}

__global__ void ScanAndWritePartSumKernelV8(const int32_t* input, int32_t* part,
                                            int32_t* output, size_t n,
                                            size_t part_num) {
    // the first 32 is used to save warp sum
    extern __shared__ int32_t shm[];
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        // store this part input to shm
        size_t index = part_i * blockDim.x + threadIdx.x;
        shm[32 + threadIdx.x] = index < n ? input[index] : 0;
        __syncthreads();
        // scan on shared memory
        ScanBlockV6(shm + 32 + threadIdx.x);
        __syncthreads();
        // write result
        if (index < n) {
            output[index] = shm[32 + threadIdx.x];
        }
        if (threadIdx.x == blockDim.x - 1) {
            part[part_i] = shm[32 + threadIdx.x];
        }
    }
}

void ScanThenFanV8(const int32_t* input, int32_t* buffer,
                   int32_t* output, size_t n) {
    size_t part_size = 1024;  // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    // use buffer[0:part_num] to save the metric of part
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    size_t shm_size = (32 + part_size) * sizeof(int32_t);
    ScanAndWritePartSumKernelV8<<<block_num, part_size, shm_size>>>(
            input, part, output, n, part_num);
    // after following step, part[i] = part_sum[0] + part_sum[1] + ... part_sum[i]
    ScanPartSumKernel<<<1, 1>>>(part, part_num);
    // make final result
    AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num);
}

// ------------------------------------------------------------------------------
__device__ __forceinline__ int32_t ScanWarpV5(int32_t val);

// ------------------------------------------------------------------------------
void ScanThenFanV9(const int32_t* input, int32_t* buffer, int32_t* output,
                   size_t n) {
    size_t part_size = 1024;  // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    // use buffer[0:part_num] to save the metric of part
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    size_t shm_size =(32 + part_size) * sizeof(int32_t);
    ScanAndWritePartSumKernelV8<<<block_num, part_size, shm_size>>>(
            input, part, output, n, part_num);
    if (part_num >= 2) {
        // after following step
        // part[i] = part_sum[0] + part_sum[1] + ... + part_sum[i]
        ScanThenFanV9(part, buffer + part_num, part, part_num);
        // make final result
        AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num);
    }
}

// ------------------------------------------------------------------------------
__global__ void ReducePartSumKernel(const int32_t* input, int32_t* part_sum,
                                    int32_t* output, size_t n, size_t part_num) {
    using BlockReduce = cub::BlockReduce<int32_t, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        size_t index = part_i * blockDim.x + threadIdx.x;
        int32_t val = index < n ? input[index] : 0;
        int32_t sum = BlockReduce(temp_storage).Sum(val);
        if (threadIdx.x == 0) {
            part_sum[part_i] = sum;
        }
        __syncthreads();
    }
}

__global__ void ScanWithBaseSum(const int32_t* input, int32_t* part_sum,
                                int32_t* output, size_t n, size_t part_num) {
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        size_t index = part_i * blockDim.x + threadIdx.x;
        int32_t val = index < n ? input[index] : 0;
        val = ScanBlockV7(val);
        __syncthreads();
        if (part_i >= 1) {
            val += part_sum[part_i - 1];
        }
        if (index < n) {
            output[index] = val;
        }
    }
}

void ReduceThenScanV1(const int32_t* input, int32_t* buffer,
                      int32_t* output, size_t n) {
    size_t part_size = 1024;  // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = std::min<size_t>(part_num, 128);
    int32_t* part_sum = buffer;  // use buffer[0:part_num]
    if (part_num >= 2) {
        ReducePartSumKernel<<<block_num, part_size>>>(input, part_sum,
                                                      output, n, part_num);
        ReduceThenScanV1(part_sum, buffer + part_num, part_sum, part_num);
    }
    ScanWithBaseSum<<<block_num, part_size, 32 * sizeof(int32_t)>>>(
            input, part_sum, output, n, part_num);
}

// ------------------------------------------------------------------------------
__global__ void ReducePartSumKernelSinglePass(const int32_t* input,
                                              int32_t* g_part_sum, size_t n,
                                              size_t part_size) {
    // this block process input[part_begin:part_end]
    size_t part_begin = blockIdx.x * part_size;
    size_t part_end = min((blockIdx.x + 1) * part_size, n);
    // part_sum
    int32_t part_sum = 0;
    for (size_t i = part_begin + threadIdx.x; i < part_end; i += blockDim.x) {
        part_sum += input[i];
    }

    using BlockReduce = cub::BlockReduce<int32_t, 1024>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    part_sum = BlockReduce(temp_storage).Sum(part_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        g_part_sum[blockIdx.x] = part_sum;
    }
}

__global__ void ScanWithBaseSumSinglePass(const int32_t* input,
                                          int32_t* g_base_sum, int32_t* output,
                                          size_t n, size_t part_size,
                                          bool debug) {
    // base sum
    __shared__ int32_t base_sum;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
            base_sum = 0;
        } else {
            base_sum = g_base_sum[blockIdx.x - 1];
        }
    }
    __syncthreads();
    // this block process input[part_begin:part_end]
    size_t part_begin = blockIdx.x * part_size;
    size_t part_end = (blockIdx.x + 1) * part_size;
    for (size_t i = part_begin + threadIdx.x; i < part_end; i += blockDim.x) {
        int32_t val = i < n ? input[i] : 0;
        val = ScanBlockV7(val);
        if (i < n) {
            output[i] = val + base_sum;
        }
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) {
            base_sum += val;
        }
        __syncthreads();
    }
}

void ReduceThenScanTwoPass(const int32_t* input, int32_t* part_sum,
                           int32_t* output, size_t n) {
    size_t part_num = 1024;
    size_t part_size = (n + part_num - 1) / part_num;
    ReducePartSumKernelSinglePass<<<part_num, 1024>>>(input, part_sum, n, part_size);
    // <<<1, 1024>>>
    ScanWithBaseSumSinglePass<<<1, 1024, 32 * sizeof(int32_t)>>>(
            part_sum, nullptr, part_sum, part_num, part_num, true);
    // <<<1024, 1024>>>
    ScanWithBaseSumSinglePass<<<part_num, 1024, 32 * sizeof(int32_t)>>>(
            input, part_sum, output, n, part_size, false);
}

// cub
void test_cub_prefix_sum(void* d_temp_storage, size_t temp_storage_bytes,
                         int32_t* input, int32_t* output, size_t n) {
    // ExclusiveSum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, input, output, n);
    // Allocate temporary storage
    checkCudaRuntime(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run exclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, input, output, n);
    // Free temporary storage
    checkCudaRuntime(cudaFree(d_temp_storage));
}

void test_ScanThenFan() {
    Timer timer;
    char str[100];
    int32_t* input;
    int32_t* buffer;
    int32_t* output;
    int32_t* output_cpu;
    const int n = 10000000;    // 1000000 4000
    size_t part_size = 1024;   // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    const size_t part_num_size = sizeof(int32_t) * part_num;
    const size_t size = sizeof(int32_t) * n;

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    checkCudaRuntime(cudaMallocManaged((void**)&input, size));
    checkCudaRuntime(cudaMallocManaged((void**)&buffer, part_num_size));
    checkCudaRuntime(cudaMallocManaged((void**)&output, size));
    checkCudaRuntime(cudaMallocManaged((void**)&output_cpu, size));
    for (int i = 0; i < n; ++i) {
        input[i] = 1;
    }
    timer.start_gpu();

    // ScanThenFan(input, buffer, output, n);
    // ScanThenFanV2(input, buffer, output, n);
    // ScanThenFanV3(input, buffer, output, n);
    // ScanThenFanV4(input, buffer, output, n);
    // ScanThenFanV5(input, buffer, output, n);
    // ScanThenFanV6(input, buffer, output, n);
    // ScanThenFanV7(input, buffer, output, n);
    // ScanThenFanV8(input, buffer, output, n);
    // ScanThenFanV9(input, buffer, output, n);

    // ReduceThenScanV1(input, buffer, output, n);
    ReduceThenScanTwoPass(input, buffer, output, n);

    // test_cub_prefix_sum(d_temp_storage, temp_storage_bytes, input, output, n);

    cudaDeviceSynchronize();

    timer.stop_gpu();
    std::sprintf(str, "PrefixSum in gpu");
    timer.duration_gpu(str);

    for (int i = 0; i < 200; ++i) {
        std::cout << output[i] << " ";
    }
    printf("\n");

    timer.start_cpu();
    PrefixSum(input, n, output_cpu);
    timer.stop_cpu();
    std::sprintf(str, "PrefixSum in cpu");
    timer.duration_cpu<Timer::ms>(str);

    bool is_false = false;
    for (int i = 0; i < n; ++i) {
        if (fabs(output[i] - output_cpu[i]) != 0) {
            is_false = true;
        }
    }
    INFO("is_false:%d", is_false);
    checkCudaRuntime(cudaFree(input));
    checkCudaRuntime(cudaFree(buffer));
    checkCudaRuntime(cudaFree(output));
}