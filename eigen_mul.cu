#include "gplex_mul.h"

#ifdef EIGEN_TEST
#include <Eigen/Dense>

using Matrix66 = Eigen::Matrix<float, 6, 6, Eigen::AutoAlign>;

__global__ void set_mem(Matrix66* a, float val, size_t N) {
  Matrix66 v = Matrix66::Constant(val);

  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    a[n] = v;
  }
}

bool check(const int N, const Matrix66* c, bool managed)
{
  const float eps = 1e-30f;
  float c0, c36;
  if (managed) {
    c0 = c[0](0,0);
    c36 = c[1](0,0);
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(c, sizeof(Matrix66)*N, device, NULL);
  } else {
    Matrix66 h[N];
    cudaMemcpy(&h, c, N*sizeof(Matrix66), cudaMemcpyDefault);
    c0 = h[0](0,0);
    c36 = h[1](0,0);
  }
  bool pass = (std::abs(c0 - c36) < eps) && (std::abs(c0 - 6.0f) < eps);
  if (!pass) {
    std::cout << "Fail check c[0]=" << c0 << " c[36]=" << c36 << std::endl;
  }
  return pass;
}

__global__ void eigen_naive_mult_kn(const Matrix66* RESTRICT a, const Matrix66* RESTRICT b, Matrix66* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    c[n] = a[n] * b[n];
  }
}

__global__ void eigen_reg_c_mult_kn(const Matrix66* RESTRICT a, const Matrix66* RESTRICT b, Matrix66* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    Matrix66 c_reg;
    c_reg = a[n] * b[n];
    c[n] = c_reg;
  }
}

__global__ void eigen_reg_mult_kn(const Matrix66* RESTRICT a, const Matrix66* RESTRICT b, Matrix66* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    Matrix66 a_reg(a[n]), b_reg(b[n]);
    Matrix66 c_reg(a_reg * b_reg);
    c[n] = c_reg;
  }
}

void eigen_run_naive_mul(int iter, bool managed)
{
  constexpr int N = Nwidth;
  constexpr int sz = sizeof(Matrix66)*N;

  Matrix66* a;
  Matrix66* b;
  Matrix66* c;

  if (managed) {
    cudaMallocManaged((void**)&a, sz);
    cudaMallocManaged((void**)&b, sz);
    cudaMallocManaged((void**)&c, sz);

    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(a, sz, device, NULL);
    cudaMemPrefetchAsync(b, sz, device, NULL);
    cudaMemPrefetchAsync(c, sz, device, NULL);
  } else {
    cudaMalloc((void**)&a, sz);
    cudaMalloc((void**)&b, sz);
    cudaMalloc((void**)&c, sz);
  }
  cudaCheckError();

  dim3 grid (((N-1)/block_size + 1), 1, 1);
  dim3 block (block_size, 1, 1);

  set_mem <<< grid, block >>> (a, 1.f , N);
  set_mem <<< grid, block >>> (b, 1.f, N);
  set_mem <<< grid, block >>> (c, 0.f, N);

  if (managed) {
    cudaMemAdvise(a, sz, cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(b, sz, cudaMemAdviseSetReadMostly, 0);
  }

  cudaCheckErrorSync();

  for (int i = 0; i < iter; ++i)
    eigen_naive_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, managed));

  for (int i = 0; i < iter; ++i)
    eigen_reg_c_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, managed));

  for (int i = 0; i < iter; ++i)
    eigen_reg_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, managed));

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaCheckErrorSync();
}
#endif
