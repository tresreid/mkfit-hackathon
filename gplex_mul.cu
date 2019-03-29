#include "gplex_mul.h"
#include "GPlex.h"

template <typename GPlex>
__global__ void set_mem(GPlex a, float val, size_t N) {

  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    for (int i = 0; i < GPlex::kSize; ++i) {
      a(n, 0, i) = val;
    }
  }
}

template <typename GPlexNM, typename GPlexMP, typename GPlexNP>
__global__ void naive_mult_kn(const RESTRICT GPlexNM a, const RESTRICT GPlexMP b, GPlexNP c, const int N)
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
__global__ void reg_c_mult_kn(const RESTRICT GPlexNM a, const RESTRICT GPlexMP b, GPlexNP c, const int N)
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
__global__ void shared_mult_kn(const RESTRICT GPlexNM a, const RESTRICT GPlexMP b, GPlexNP c, const int N)
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
__global__ void reg_mult_kn(const RESTRICT GPlexNM a, const RESTRICT GPlexMP b, GPlexNP c, const int N)
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

bool check(int N, GPlexLL c, bool managed)
{
  const float eps = 1e-30;
  float c0, c36;
  if (managed) {
    c0 = c(0,0,0);
    c36 = c(1,0,0);
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(c.Ptr(), N*sizeof(float)*GPlexLL::kSize, device, NULL);
  } else {
    GPlexBLL h[(N+NN-1)/NN];
    c.copyToHost(h[0]);
    c0 = h[0].At(0,0,0);
    c36 = h[1].At(1,0,0);
  }
  bool pass = (std::abs(c0 - c36) < eps) && (std::abs(c0 - 6.0f) < eps);
  if (!pass) {
    std::cout << "Fail check c[0]=" << c0 << " c[36]=" << c36 << std::endl;
  }
  return pass;
}


void run_naive_mul(int iter, bool managed)
{
  constexpr int N = Nwidth;
  GPlexLL a, b, c;
  constexpr int sz = N*sizeof(float)*GPlexLL::kSize;

  if (managed) {
    a.allocateManaged(N);
    b.allocateManaged(N);
    c.allocateManaged(N);

    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(a.Ptr(), sz, device, NULL);
    cudaMemPrefetchAsync(b.Ptr(), sz, device, NULL);
    cudaMemPrefetchAsync(c.Ptr(), sz, device, NULL);
  } else {
    a.allocate(N);
    b.allocate(N);
    c.allocate(N);
  }
  cudaCheckError();

  dim3 grid (((N-1)/block_size + 1), 1, 1);
  dim3 block (block_size, 1, 1);

  set_mem <<< grid, block >>> (a, 1.f , N);
  set_mem <<< grid, block >>> (b, 1.f, N);
  set_mem <<< grid, block >>> (c, 0.f, N);

  if (managed) {
    cudaMemAdvise(a.Ptr(), sz, cudaMemAdviseSetReadMostly, 0);
    cudaMemAdvise(b.Ptr(), sz, cudaMemAdviseSetReadMostly, 0);
  }

  cudaCheckErrorSync();

  for (int i = 0; i < iter; ++i) {
    set_mem <<< grid, block >>> (c, 0.f, N);
    naive_mult_kn <<< grid, block >>> (a, b, c, N);
  }
  cudaCheckErrorSync();
  assert(check(N, c, managed));

  for (int i = 0; i < iter; ++i)
    reg_c_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, managed));

  for (int i = 0; i < iter; ++i)
    shared_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, managed));

  for (int i = 0; i < iter; ++i)
    reg_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, managed));

  a.free();
  b.free();
  c.free();
  cudaCheckErrorSync();
}
