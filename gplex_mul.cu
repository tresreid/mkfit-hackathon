#include "gpu_utils.h"

#include <vector>
#include <iostream>

constexpr int block_size = 64;


__global__ void raw_naive_mult_kn(const float* a,
    const float* b, float* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        for (int k = 0; k < 6; ++k) {
          c[n + N*(i + 6*j)] += a[n + N*(i + 6*k)] * b[n + N*(k + 6*j)];
        }
      }
    }
  }
}

__global__ void raw_reg_c_mult_kn(const float* a, const float* b, 
    float* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        float c_tmp = 0;
        for (int k = 0; k < 6; ++k) {
          c_tmp += a[n + N*(i + 6*k)] * b[n + N*(k + 6*j)];
        }
        c[n + N*(i + 6*j)] += c_tmp;
      }
    }
  }
}

__global__ void raw_shared_mult_kn(const float* a, const float* b, float* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    int tix = threadIdx.x;

    __shared__ float sh_a[36][block_size];
    __shared__ float sh_b[36][block_size];

    for (int i = 0; i < 36; ++i) {
      sh_a[i][tix] = a[n + 36*i];
    }
    for (int i = 0; i < 36; ++i) {
      sh_b[i][tix] = b[n + 36*i];
    }
    __syncthreads();

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        float c_tmp = 0;
        for (int k = 0; k < 6; ++k) {
          /*c_tmp += a(n, i, k) * b(n, k, j);*/
          c_tmp += sh_a[k + 6* i][tix] 
            * sh_b[j + 6 * k][tix];
          /*c_tmp += sh_a[0][tix] ;*/
            /** sh_b[j + GPlexMP::kCols * k][tix];*/
        }
        c[n + N*(i + 6*j)] += c_tmp;
      }
    }
  }
}

__global__ void raw_reg_mult_kn(const float* a, const float* b, float* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
      n < N;
      n += blockDim.x * gridDim.x) {

    float reg_a[36];
    float reg_b[36];

    for (int i = 0; i < 36; ++i) {
      reg_a[i] = a[n + 36*i];
    }
    for (int i = 0; i < 36; ++i) {
      reg_b[i] = b[n + 36*i];
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        float c_tmp = 0;
        for (int k = 0; k < 6; ++k) {
          c_tmp += reg_a[i+6*k] * reg_b[k+6+j];
        }
        c[n + N*(i + 6*j)] += c_tmp;
      }
    }
  }
}


void raw_run_naive_mul(int N)
{
  /*constexpr int N = 3200000;//10000000;*/
  /*constexpr int N = 100000;*/
  float* a;
  float* b;
  float* c;

  cudaMalloc((void**)&a, 36*N*sizeof(float));
  cudaMalloc((void**)&b, 36*N*sizeof(float));
  cudaMalloc((void**)&c, 36*N*sizeof(float));
  cudaCheckErrorSync();


  dim3 grid (((N-1)/block_size + 1), 1, 1);
  dim3 block (block_size, 1, 1);

  cudaMemset(a, 1, 36*N*sizeof(float));
  cudaMemset(b, 1, 36*N*sizeof(float));
  cudaMemset(c, 0, 36*N*sizeof(float));
  cudaCheckErrorSync();

  raw_naive_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  raw_reg_c_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  raw_shared_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  raw_reg_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();

  std::vector<float> h_c (N);
  if (h_c[0] == 42)
    std::cout << h_c[0] << std::endl;

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaCheckErrorSync();
}

