//
// Created by ros on 3/11/24.
//
#include <cuda_runtime.h>
#include <stdio.h>
#include <random>

#define checkCudaKernel(...)                                                                         \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        printf("launch failed: %s\n", cudaGetErrorString(cudaStatus));                               \
    }} while(0);

#define checkCudaRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)
bool check_runtime(cudaError_t e, const char* call, int line, const char *file){
    if (e != cudaSuccess) {
        printf("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n",
               call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

// B=2, T=10, N=128, D=32, H=8, E=256
const int B = 2;
const int H = 8;
const int T = 10;
const int D = 32;
const int BHTD3 = B * H * T * D * 3;

// in:[B, T, 3, H, D]->out:[3, B, H, T, D]用cuda实现
// gird:[T, B]
// block:[H * D]
template<typename dtype>
__global__ void SpiltQKVAndAddBiasKernel(const dtype* mm_qkv, const dtype* bias, dtype* qkv,
                                         const int B, const int H, const int T, const int D) {
    int B_i = blockIdx.y;
    int T_i = blockIdx.x;
    int tid = threadIdx.x;
    int HD = H * D;
    int BHTD = B * H * T * D;
    int bias_index = tid;
    int base_in = (B_i * T + T_i) * 3 * H * D;
    int out_index = B_i * H * T * D + (tid / D) * T * D + T_i * D + (tid % D);
    qkv[out_index] = mm_qkv[base_in + bias_index] + bias[bias_index];
    bias_index += HD;
    out_index += BHTD;
    qkv[out_index] = mm_qkv[base_in + bias_index] + bias[bias_index];
    bias_index += HD;
    out_index += BHTD;
    qkv[out_index] = mm_qkv[base_in + bias_index] + bias[bias_index];
}

int main () {
    size_t in_size = B * T * 3 * H * D * sizeof(float);
    size_t out_size = 3 * B * H * T * D * sizeof(float);
    size_t bias_size = 3 * H * D * sizeof(float);
    float* in;
    float* out;
    float* bias;
    in = new float[in_size];
    out = new float[out_size];
    bias = new float[bias_size];
    std::default_random_engine e;
    std::normal_distribution<float> u(0,1); // 均值为0，标准差为1
    e.seed(0);

    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < T; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int m = 0; m < H; ++m) {
                    for (int n = 0; n < D; ++n) {
                        in[i * (T * 3 * H * D) + j * (3 * H * D) + k * (H * D) + m * D + n] = u(e);
                    }
                }
            }
        }
    }
    // for (int i = 0; i < B; ++i) {
    //     for (int j = 0; j < T; ++j) {
    //         for (int k = 0; k < 3; ++k) {
    //             for (int m = 0; m < H; ++m) {
    //                 for (int n = 0; n < D; ++n) {
    //                     printf("%f ", in[i * (T * 3 * H * D) + j * (3 * H * D) + k * (H * D) + m * D + n]);
    //                 }
    //             }
    //         }
    //     }
    // }
    // for (int i = 0; i < 500; ++i) {
    //     printf("%f ", in[i]);
    // }
    // printf("\n");

    float* in_gpu;
    float* bias_gpu;
    float* out_gpu;
    checkCudaRuntime(cudaMalloc((void**)&in_gpu, in_size));
    checkCudaRuntime(cudaMalloc((void**)&bias_gpu, bias_size));
    checkCudaRuntime(cudaMalloc((void**)&out_gpu, out_size));
    checkCudaRuntime(cudaMemcpy(in_gpu, in, in_size, cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(bias_gpu, bias, bias_size, cudaMemcpyHostToDevice));

    dim3 grid(T, B);
    dim3 block(H * D);

    checkCudaKernel(SpiltQKVAndAddBiasKernel<float><<<grid, block>>>(in_gpu, bias_gpu, out_gpu, B, H, T, D));

    checkCudaRuntime(cudaMemcpy(out, out_gpu, out_size, cudaMemcpyDeviceToHost));

    printf("-------------------------------------------------------------------\n");
    // 3 B H T D
    // for (int i = 0; i < 3; ++i) {
    //     for (int j = 0; j < B; ++j) {
    //         for (int k = 0; k < H; ++k) {
    //             for (int m = 0; m < T; ++m) {
    //                 for (int n = 0; n < D; ++n) {
    //                     printf("%f ", out[i * (B * H * T * D) + j * (H * T * D) + k * (T * D) + m * D + n]);
    //                 }
    //             }
    //         }
    //     }
    // }
    std::vector<int> diff_index = {};
    diff_index.reserve(BHTD3);
    for (int i = 0; i < BHTD3; ++i) {
        // printf("%f ", out[i]);
        if (fabs(in[i] - out[i]) != 0) {
            diff_index.push_back(i);
        }
    }
    for (int i = 0; i < diff_index.size(); ++i) {
        printf("%d ", diff_index.at(i));
    }
    printf("\n");
    printf("%zu\n", diff_index.size());

    cudaFree(in_gpu);
    cudaFree(out_gpu);
    delete[] in;
    delete[] bias;
    delete[] out;
}