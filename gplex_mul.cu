#define CUB_STDERR
#include "gpu_utils.h"
#include "gplex_mul.h"
#include "cuda_profiler_api.h"
#include "GPlex.h"

#include <vector>
#include <iostream>

constexpr int block_size = 64;

template <typename GPlex>
__global__ void set_mem(GPlex a, float val,size_t N) {

  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    for (int i = 0; i < GPlex::kSize; ++i) {
      a(n, 0, i) = val;
    }
  }
}

template <typename GPlexNM, typename GPlexMP, typename GPlexNP>
__global__ void naive_mult_kn(const __restrict__ GPlexNM a, const __restrict__ GPlexMP b, GPlexNP c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {


    for (int i = 0; i < GPlexNM::kRows; ++i) {
      for (int j = 0; j < GPlexMP::kCols; ++j) {
        for (int k = 0; k < GPlexNM::kCols; ++k) {
          c(n, i, j) += a(n, i, k) * b(n, k, j);
        }
      }
    }
  }
}

template <typename GPlexNM, typename GPlexMP, typename GPlexNP>
__global__ void reg_c_mult_kn(const __restrict__ GPlexNM a, const __restrict__ GPlexMP b, GPlexNP c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    for (int i = 0; i < GPlexNM::kRows; ++i) {
      for (int j = 0; j < GPlexMP::kCols; ++j) {
        float c_tmp = 0;
        for (int k = 0; k < GPlexNM::kCols; ++k) {
          c_tmp += a(n, i, k) * b(n, k, j);
        }
        c(n, i, j) = c_tmp;
      }
    }
  }
}

template <typename GPlexNM, typename GPlexMP, typename GPlexNP>
__global__ void shared_mult_kn(const __restrict__ GPlexNM a, const __restrict__ GPlexMP b, GPlexNP c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    int tix = threadIdx.x;

    __shared__ float sh_a[GPlexNM::kSize][block_size];
    __shared__ float sh_b[GPlexMP::kSize][block_size];

    for (int i = 0; i < GPlexNM::kSize; ++i) {
      sh_a[i][tix] = a(n, 0, i);
    }
    for (int i = 0; i < GPlexNM::kSize; ++i) {
      sh_b[i][tix] = b(n, 0, i);
    }
    __syncthreads();

    for (int i = 0; i < GPlexNM::kRows; ++i) {
      for (int j = 0; j < GPlexMP::kCols; ++j) {
        float c_tmp = 0;
        for (int k = 0; k < GPlexNM::kCols; ++k) {
          /*c_tmp += a(n, i, k) * b(n, k, j);*/
          c_tmp += sh_a[k + GPlexNM::kCols * i][tix] 
            * sh_b[j + GPlexMP::kCols * k][tix];
          /*c_tmp += sh_a[0][tix] ;*/
            /** sh_b[j + GPlexMP::kCols * k][tix];*/
        }
        c(n, i, j) = c_tmp;
      }
    }
  }
}

template <typename GPlexNM, typename GPlexMP, typename GPlexNP>
__global__ void reg_mult_kn(const __restrict__ GPlexNM a, const __restrict__ GPlexMP b, GPlexNP c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    GPlexRegLL reg_a;
    GPlexRegLL reg_b;

    for (int i = 0; i < GPlexNM::kSize; ++i) {
      reg_a[i] = a(n, 0, i);
    }
    for (int i = 0; i < GPlexMP::kSize; ++i) {
      reg_b[i] = b(n, 0, i);
    }

    for (int i = 0; i < GPlexNM::kRows; ++i) {
      for (int j = 0; j < GPlexMP::kCols; ++j) {
        float c_tmp = 0;
        for (int k = 0; k < GPlexNM::kCols; ++k) {
          c_tmp += reg_a(n, i, k) * reg_b(n, k, j);
        }
        c(n, i, j) = c_tmp;
      }
    }
  }
}


void run_naive_mul(int N, int iter, bool pauseProf)
{
  /*constexpr int N = 3200000;//10000000;*/
  /*constexpr int N = 100000;*/
  GPlexLL a, b, c;

  a.allocate(N);
  b.allocate(N);
  c.allocate(N);
  cudaCheckErrorSync();

  dim3 grid (((N-1)/block_size + 1), 1, 1);
  dim3 block (block_size, 1, 1);

  set_mem <<< grid, block >>> (a, 1.f , N);
  set_mem <<< grid, block >>> (b, 1.f, N);
  set_mem <<< grid, block >>> (c, 0.f, N);
  cudaCheckErrorSync();

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    naive_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    reg_c_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    shared_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    reg_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  std::vector<float> h_c (N);
  if (h_c[0] == 42)
    std::cout << h_c[0] << std::endl;

  a.free();
  b.free();
  c.free();
  cudaCheckErrorSync();
}

__global__ void raw_naive_mult_kn(const float* __restrict__ a,
    const float* __restrict__ b, float* __restrict__ c, const int N)
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

__global__ void raw_reg_c_mult_kn(const float* __restrict__ a, const float* __restrict__ b, 
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
        c[n + N*(i + 6*j)] = c_tmp;
      }
    }
  }
}

__global__ void raw_shared_mult_kn(const float* __restrict__ a, const float* __restrict__ b, float* c, const int N)
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
        c[n + N*(i + 6*j)] = c_tmp;
      }
    }
  }
}

__global__ void raw_reg_mult_kn(const float* __restrict__ a, const float* __restrict__ b, float* c, const int N)
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
        c[n + N*(i + 6*j)] = c_tmp;
      }
    }
  }
}


void raw_run_naive_mul(int N, int iter, bool pauseProf)
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

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    raw_naive_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    raw_reg_c_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    raw_shared_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    raw_reg_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  std::vector<float> h_c (N);
  if (h_c[0] == 42)
    std::cout << h_c[0] << std::endl;

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaCheckErrorSync();
}

#ifdef EIGEN_TEST
#include <Eigen/Dense>

using Matrix66 = Eigen::Matrix<float, 6, 6, Eigen::AutoAlign>;

__global__ void eigen_reg_c_mult_kn(const Matrix66* __restrict__ a, const Matrix66* __restrict__ b, Matrix66* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    Matrix66 c_reg;
    c_reg = a[n] * b[n];
    c[n] = c_reg;
  }
}

__global__ void eigen_reg_mult_kn(const Matrix66* __restrict__ a, const Matrix66* __restrict__ b, Matrix66* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    Matrix66 a_reg(a[n]), b_reg(b[n]);
    Matrix66 c_reg(a_reg * b_reg);
    c[n] = c_reg;
  }
}

void eigen_run_naive_mul(int N, int iter, bool pauseProf)
{
  /*constexpr int N = 3200000;//10000000;*/
  /*constexpr int N = 100000;*/
  Matrix66* a;
  Matrix66* b;
  Matrix66* c;

  cudaMalloc((void**)&a, sizeof(Matrix66)*N);
  cudaMalloc((void**)&b, sizeof(Matrix66)*N);
  cudaMalloc((void**)&c, sizeof(Matrix66)*N);
  cudaCheckErrorSync();


  dim3 grid (((N-1)/block_size + 1), 1, 1);
  dim3 block (block_size, 1, 1);

  Matrix66 zero = Matrix66::Zero();
  Matrix66 one = Matrix66::Constant(1);

  for (int i = 0; i < N; ++i) {
    cudaMemcpy(a+i, &one, sizeof(Matrix66), cudaMemcpyDefault);
    cudaMemcpy(b+i, &one, sizeof(Matrix66), cudaMemcpyDefault);
    cudaMemcpy(c+i, &zero, sizeof(Matrix66), cudaMemcpyDefault);
  }
  cudaCheckErrorSync();

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    eigen_reg_c_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  if (pauseProf) cudaProfilerStart();
  for (int i = 0; i < iter; ++i)
    eigen_reg_mult_kn <<< grid, block >>> (a, b, c, N);
  if (pauseProf) cudaProfilerStop();
  cudaCheckErrorSync();

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaCheckErrorSync();
}
#endif

