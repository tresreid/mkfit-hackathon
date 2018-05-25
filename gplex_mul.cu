#define CUB_STDERR
#include "gpu_utils.h"
#include "gplex_mul.h"
#include "cuda_profiler_api.h"
#include "GPlex.h"

#include <iostream>
#include <cassert>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
#define RESTRICT /* */
#else
#define RESTRICT __restrict__
#endif

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

bool check(int N, GPlexLL c)
{
  MPlexLL h[N];
  const float eps = 1e-30;
  c.copyToHost(h[0]);
  return (std::abs(h[0].At(0,0,0) - h[1].At(1,0,0)) < eps) && (std::abs(h[0].At(0,0,0) - 6.0f) < eps);
}


void run_naive_mul(int N, int iter)
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

  for (int i = 0; i < iter; ++i)
    naive_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c));

  for (int i = 0; i < iter; ++i)
    reg_c_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c));

  for (int i = 0; i < iter; ++i)
    shared_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c));

  for (int i = 0; i < iter; ++i)
    reg_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c));

  a.free();
  b.free();
  c.free();
  cudaCheckErrorSync();
}

__global__ void raw_naive_mult_kn(const float* RESTRICT a,
    const float* RESTRICT b, float* RESTRICT c, const int N)
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

__global__ void raw_reg_c_mult_kn(const float* RESTRICT a, const float* RESTRICT b, 
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

/* Try to do all from the registers; the code is adjusted to get to 100% FLOP efficiency */
__global__ void raw_regall_c_mult_loop_kn(const float* RESTRICT a, const float* RESTRICT b,
                                          float* c, const int N)
{

  float a_ar[36];
  float b_ar[36];
  float c_ar[36];
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {
    for (int i = 0; i < 36; ++i){
      const int idx = n + N*i;
      a_ar[i] = a[idx];
      b_ar[i] = b[idx];
      c_ar[i] = 0.f;
    }
  }

  int nN = 10000;
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    for (int oLoop = 0; oLoop< nN; ++oLoop){

      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          float c_tmp = 0.f;
          for (int k = 0; k < 6; ++k) {
            c_tmp += a_ar[i + 6*k] * b_ar[k + 6*j];
          }
          c_ar[i + 6*j] += c_tmp*3.4f;
        }
      }
    }
  }//oLoop< nN; ++oLoop){

  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {
    for (int i = 0; i < 36; ++i){
      const int idx = n + N*i;
      c[idx] = c_ar[i];
    }
  }

}


__global__ void raw_reg_c_mult_loop_kn(const float* RESTRICT const a, const float* RESTRICT const b, 
    float* c, const int N)
{

  int nN = 1000;
  for (int oLoop = 0; oLoop< nN; ++oLoop){
    for (int n = threadIdx.x + blockIdx.x * blockDim.x;
         n < N;
         n += blockDim.x * gridDim.x) {
      
      float a_ar[36];
      float b_ar[36];
      for (int i = 0; i < 36; ++i){
        const int idx = n + N*i;
        a_ar[i] = a[idx];
        b_ar[i] = b[idx];
      }
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          float c_tmp = 0.f;
          for (int k = 0; k < 6; ++k) {
            c_tmp += a_ar[i + 6*k] * b_ar[k + 6*j];
          }
          c[n + N*(i + 6*j)] = c_tmp;
        }
      }
    }
  }//oLoop< nN; ++oLoop){
}

/** like raw_reg_c_mult_loop_kn but with vectorized loads and 
    therefore a different memory layout of the matrix */
__global__ void raw_reg_c_mult_loop_kn_vl(const float* RESTRICT const a, const float* RESTRICT const b, 
    float* c, const int N)
{

  int nN = 1000;
  for (int oLoop = 0; oLoop< nN; ++oLoop){
    for (int n = threadIdx.x + blockIdx.x * blockDim.x;
         n < N;
         n += blockDim.x * gridDim.x) {
      
      float a_ar[36];
      float b_ar[36];
      for (int i = 0; i < 9; ++i){
        const int idx = n + N*i;
        reinterpret_cast<float4*>(a_ar)[i] = reinterpret_cast<const float4*>(a)[idx];
        reinterpret_cast<float4*>(b_ar)[i] = reinterpret_cast<const float4*>(b)[idx];
      }
      
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          float c_tmp = 0.f;
          for (int k = 0; k < 6; ++k) {
            c_tmp += a_ar[i + 6*k] * b_ar[k + 6*j];
          }
          c[n + N*(i + 6*j)] = c_tmp;
        }
      }
    }
  }//oLoop< nN; ++oLoop){
}

/** like raw_reg_c_mult_loop_kn_vl but also using 
    vectorized stores */
__global__ void raw_reg_c_mult_loop_kn_vl_vs(const float* RESTRICT const a, const float* RESTRICT const b, 
    float* c, const int N)
{
  int nN = 1000;
  for (int oLoop = 0; oLoop< nN; ++oLoop){
    for (int n = threadIdx.x + blockIdx.x * blockDim.x;
         n < N;
         n += blockDim.x * gridDim.x) {
      
      float a_ar[36];
      float b_ar[36];
      float c_ar[36];
      for (int i = 0; i < 9; ++i){
        const int idx = n + N*i;
        reinterpret_cast<float4*>(a_ar)[i] = reinterpret_cast<const float4*>(a)[idx];
        reinterpret_cast<float4*>(b_ar)[i] = reinterpret_cast<const float4*>(b)[idx];
      }
      
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          float c_tmp = 0.f;
          for (int k = 0; k < 6; ++k) {
            c_tmp += a_ar[i + 6*k] * b_ar[k + 6*j];
          }
	  c_ar[i + 6 * j] = c_tmp;
        }
      }

      // write back
      for (int i = 0; i < 9; ++i){
	const int idx = n + N*i;
	reinterpret_cast<float4*>(c)[idx] = reinterpret_cast<const float4*>(c_ar)[i];
      }
    }
  }//oLoop< nN; ++oLoop){
}


/** like raw_reg_c_mult_loop_kn_vl but assuming that the matrix B is 
    transposed thus making the index calculation for matrix A
    and B the same */
__global__ void raw_reg_c_mult_loop_kn_vl_transp(const float* RESTRICT const a, const float* RESTRICT const b, 
    float* c, const int N)
{

  int nN = 1000;
  for (int oLoop = 0; oLoop< nN; ++oLoop){
    for (int n = threadIdx.x + blockIdx.x * blockDim.x;
         n < N;
         n += blockDim.x * gridDim.x) {
      
      float a_ar[36];
      float b_ar[36];
      for (int i = 0; i < 9; ++i){
        const int idx = n + N*i;
        reinterpret_cast<float4*>(a_ar)[i] = reinterpret_cast<const float4*>(a)[idx];
        reinterpret_cast<float4*>(b_ar)[i] = reinterpret_cast<const float4*>(b)[idx];
      }
      
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          float c_tmp = 0.f;
          for (int k = 0; k < 36; k += 6) {
            c_tmp += a_ar[i + k] * b_ar[j + k];
          }
	  c[n + N *(i + 6 * j)] = c_tmp;
        }
      }
    }
  }//oLoop< nN; ++oLoop){
}


__global__ void raw_reg_c_mult_loop_unroll_kn(const float* RESTRICT const a, const float* RESTRICT const b, 
    float* c, const int N)
{
  int nN = 1000;
  for (int oLoop = 0; oLoop< nN; ++oLoop){
    for (int n = threadIdx.x + blockIdx.x * blockDim.x;
         n < N;
         n += blockDim.x * gridDim.x) {
      float c_temp;
      float a_00 = a[n + N*0];
      float b_00 = b[n + N*0];
      c_temp =  a_00  * b_00;
      float a_01 = a[n + N*6];
      float b_01 = b[n + N*6];
      float a_10 = a[n + N*1];
      float b_10 = b[n + N*1];
      c_temp +=  a_01  * b_10;
      float a_02 = a[n + N*12];
      float b_02 = b[n + N*12];
      float a_20 = a[n + N*2];
      float b_20 = b[n + N*2];
      c_temp +=  a_02  * b_20;
      float a_03 = a[n + N*18];
      float b_03 = b[n + N*18];
      float a_30 = a[n + N*3];
      float b_30 = b[n + N*3];
      c_temp +=  a_03  * b_30;
      float a_04 = a[n + N*24];
      float b_04 = b[n + N*24];
      float a_40 = a[n + N*4];
      float b_40 = b[n + N*4];
      c_temp +=  a_04  * b_40;
      float a_05 = a[n + N*30];
      float b_05 = b[n + N*30];
      float a_50 = a[n + N*5];
      float b_50 = b[n + N*5];
      c_temp +=  a_05  * b_50;
      c[n + N*0 ] = c_temp;
      c_temp =  a_00  * b_01;
      float a_11 = a[n + N*7];
      float b_11 = b[n + N*7];
      c_temp +=  a_01  * b_11;
      float a_21 = a[n + N*8];
      float b_21 = b[n + N*8];
      c_temp +=  a_02  * b_21;
      float a_31 = a[n + N*9];
      float b_31 = b[n + N*9];
      c_temp +=  a_03  * b_31;
      float a_41 = a[n + N*10];
      float b_41 = b[n + N*10];
      c_temp +=  a_04  * b_41;
      float a_51 = a[n + N*11];
      float b_51 = b[n + N*11];
      c_temp +=  a_05  * b_51;
      c[n + N*6 ] = c_temp;
      c_temp =  a_00  * b_02;
      float a_12 = a[n + N*13];
      float b_12 = b[n + N*13];
      c_temp +=  a_01  * b_12;
      float a_22 = a[n + N*14];
      float b_22 = b[n + N*14];
      c_temp +=  a_02  * b_22;
      float a_32 = a[n + N*15];
      float b_32 = b[n + N*15];
      c_temp +=  a_03  * b_32;
      float a_42 = a[n + N*16];
      float b_42 = b[n + N*16];
      c_temp +=  a_04  * b_42;
      float a_52 = a[n + N*17];
      float b_52 = b[n + N*17];
      c_temp +=  a_05  * b_52;
      c[n + N*12 ] = c_temp;
      c_temp =  a_00  * b_03;
      float a_13 = a[n + N*19];
      float b_13 = b[n + N*19];
      c_temp +=  a_01  * b_13;
      float a_23 = a[n + N*20];
      float b_23 = b[n + N*20];
      c_temp +=  a_02  * b_23;
      float a_33 = a[n + N*21];
      float b_33 = b[n + N*21];
      c_temp +=  a_03  * b_33;
      float a_43 = a[n + N*22];
      float b_43 = b[n + N*22];
      c_temp +=  a_04  * b_43;
      float a_53 = a[n + N*23];
      float b_53 = b[n + N*23];
      c_temp +=  a_05  * b_53;
      c[n + N*18 ] = c_temp;
      c_temp =  a_00  * b_04;
      float a_14 = a[n + N*25];
      float b_14 = b[n + N*25];
      c_temp +=  a_01  * b_14;
      float a_24 = a[n + N*26];
      float b_24 = b[n + N*26];
      c_temp +=  a_02  * b_24;
      float a_34 = a[n + N*27];
      float b_34 = b[n + N*27];
      c_temp +=  a_03  * b_34;
      float a_44 = a[n + N*28];
      float b_44 = b[n + N*28];
      c_temp +=  a_04  * b_44;
      float a_54 = a[n + N*29];
      float b_54 = b[n + N*29];
      c_temp +=  a_05  * b_54;
      c[n + N*24 ] = c_temp;
      c_temp =  a_00  * b_05;
      float a_15 = a[n + N*31];
      float b_15 = b[n + N*31];
      c_temp +=  a_01  * b_15;
      float a_25 = a[n + N*32];
      float b_25 = b[n + N*32];
      c_temp +=  a_02  * b_25;
      float a_35 = a[n + N*33];
      float b_35 = b[n + N*33];
      c_temp +=  a_03  * b_35;
      float a_45 = a[n + N*34];
      float b_45 = b[n + N*34];
      c_temp +=  a_04  * b_45;
      float a_55 = a[n + N*35];
      float b_55 = b[n + N*35];
      c_temp +=  a_05  * b_55;
      c[n + N*30 ] = c_temp;
      c_temp =  a_10  * b_00;
      c_temp +=  a_11  * b_10;
      c_temp +=  a_12  * b_20;
      c_temp +=  a_13  * b_30;
      c_temp +=  a_14  * b_40;
      c_temp +=  a_15  * b_50;
      c[n + N*1 ] = c_temp;
      c_temp =  a_10  * b_01;
      c_temp +=  a_11  * b_11;
      c_temp +=  a_12  * b_21;
      c_temp +=  a_13  * b_31;
      c_temp +=  a_14  * b_41;
      c_temp +=  a_15  * b_51;
      c[n + N*7 ] = c_temp;
      c_temp =  a_10  * b_02;
      c_temp +=  a_11  * b_12;
      c_temp +=  a_12  * b_22;
      c_temp +=  a_13  * b_32;
      c_temp +=  a_14  * b_42;
      c_temp +=  a_15  * b_52;
      c[n + N*13 ] = c_temp;
      c_temp =  a_10  * b_03;
      c_temp +=  a_11  * b_13;
      c_temp +=  a_12  * b_23;
      c_temp +=  a_13  * b_33;
      c_temp +=  a_14  * b_43;
      c_temp +=  a_15  * b_53;
      c[n + N*19 ] = c_temp;
      c_temp =  a_10  * b_04;
      c_temp +=  a_11  * b_14;
      c_temp +=  a_12  * b_24;
      c_temp +=  a_13  * b_34;
      c_temp +=  a_14  * b_44;
      c_temp +=  a_15  * b_54;
      c[n + N*25 ] = c_temp;
      c_temp =  a_10  * b_05;
      c_temp +=  a_11  * b_15;
      c_temp +=  a_12  * b_25;
      c_temp +=  a_13  * b_35;
      c_temp +=  a_14  * b_45;
      c_temp +=  a_15  * b_55;
      c[n + N*31 ] = c_temp;
      c_temp =  a_20  * b_00;
      c_temp +=  a_21  * b_10;
      c_temp +=  a_22  * b_20;
      c_temp +=  a_23  * b_30;
      c_temp +=  a_24  * b_40;
      c_temp +=  a_25  * b_50;
      c[n + N*2 ] = c_temp;
      c_temp =  a_20  * b_01;
      c_temp +=  a_21  * b_11;
      c_temp +=  a_22  * b_21;
      c_temp +=  a_23  * b_31;
      c_temp +=  a_24  * b_41;
      c_temp +=  a_25  * b_51;
      c[n + N*8 ] = c_temp;
      c_temp =  a_20  * b_02;
      c_temp +=  a_21  * b_12;
      c_temp +=  a_22  * b_22;
      c_temp +=  a_23  * b_32;
      c_temp +=  a_24  * b_42;
      c_temp +=  a_25  * b_52;
      c[n + N*14 ] = c_temp;
      c_temp =  a_20  * b_03;
      c_temp +=  a_21  * b_13;
      c_temp +=  a_22  * b_23;
      c_temp +=  a_23  * b_33;
      c_temp +=  a_24  * b_43;
      c_temp +=  a_25  * b_53;
      c[n + N*20 ] = c_temp;
      c_temp =  a_20  * b_04;
      c_temp +=  a_21  * b_14;
      c_temp +=  a_22  * b_24;
      c_temp +=  a_23  * b_34;
      c_temp +=  a_24  * b_44;
      c_temp +=  a_25  * b_54;
      c[n + N*26 ] = c_temp;
      c_temp =  a_20  * b_05;
      c_temp +=  a_21  * b_15;
      c_temp +=  a_22  * b_25;
      c_temp +=  a_23  * b_35;
      c_temp +=  a_24  * b_45;
      c_temp +=  a_25  * b_55;
      c[n + N*32 ] = c_temp;
      c_temp =  a_30  * b_00;
      c_temp +=  a_31  * b_10;
      c_temp +=  a_32  * b_20;
      c_temp +=  a_33  * b_30;
      c_temp +=  a_34  * b_40;
      c_temp +=  a_35  * b_50;
      c[n + N*3 ] = c_temp;
      c_temp =  a_30  * b_01;
      c_temp +=  a_31  * b_11;
      c_temp +=  a_32  * b_21;
      c_temp +=  a_33  * b_31;
      c_temp +=  a_34  * b_41;
      c_temp +=  a_35  * b_51;
      c[n + N*9 ] = c_temp;
      c_temp =  a_30  * b_02;
      c_temp +=  a_31  * b_12;
      c_temp +=  a_32  * b_22;
      c_temp +=  a_33  * b_32;
      c_temp +=  a_34  * b_42;
      c_temp +=  a_35  * b_52;
      c[n + N*15 ] = c_temp;
      c_temp =  a_30  * b_03;
      c_temp +=  a_31  * b_13;
      c_temp +=  a_32  * b_23;
      c_temp +=  a_33  * b_33;
      c_temp +=  a_34  * b_43;
      c_temp +=  a_35  * b_53;
      c[n + N*21 ] = c_temp;
      c_temp =  a_30  * b_04;
      c_temp +=  a_31  * b_14;
      c_temp +=  a_32  * b_24;
      c_temp +=  a_33  * b_34;
      c_temp +=  a_34  * b_44;
      c_temp +=  a_35  * b_54;
      c[n + N*27 ] = c_temp;
      c_temp =  a_30  * b_05;
      c_temp +=  a_31  * b_15;
      c_temp +=  a_32  * b_25;
      c_temp +=  a_33  * b_35;
      c_temp +=  a_34  * b_45;
      c_temp +=  a_35  * b_55;
      c[n + N*33 ] = c_temp;
      c_temp =  a_40  * b_00;
      c_temp +=  a_41  * b_10;
      c_temp +=  a_42  * b_20;
      c_temp +=  a_43  * b_30;
      c_temp +=  a_44  * b_40;
      c_temp +=  a_45  * b_50;
      c[n + N*4 ] = c_temp;
      c_temp =  a_40  * b_01;
      c_temp +=  a_41  * b_11;
      c_temp +=  a_42  * b_21;
      c_temp +=  a_43  * b_31;
      c_temp +=  a_44  * b_41;
      c_temp +=  a_45  * b_51;
      c[n + N*10 ] = c_temp;
      c_temp =  a_40  * b_02;
      c_temp +=  a_41  * b_12;
      c_temp +=  a_42  * b_22;
      c_temp +=  a_43  * b_32;
      c_temp +=  a_44  * b_42;
      c_temp +=  a_45  * b_52;
      c[n + N*16 ] = c_temp;
      c_temp =  a_40  * b_03;
      c_temp +=  a_41  * b_13;
      c_temp +=  a_42  * b_23;
      c_temp +=  a_43  * b_33;
      c_temp +=  a_44  * b_43;
      c_temp +=  a_45  * b_53;
      c[n + N*22 ] = c_temp;
      c_temp =  a_40  * b_04;
      c_temp +=  a_41  * b_14;
      c_temp +=  a_42  * b_24;
      c_temp +=  a_43  * b_34;
      c_temp +=  a_44  * b_44;
      c_temp +=  a_45  * b_54;
      c[n + N*28 ] = c_temp;
      c_temp =  a_40  * b_05;
      c_temp +=  a_41  * b_15;
      c_temp +=  a_42  * b_25;
      c_temp +=  a_43  * b_35;
      c_temp +=  a_44  * b_45;
      c_temp +=  a_45  * b_55;
      c[n + N*34 ] = c_temp;
      c_temp =  a_50  * b_00;
      c_temp +=  a_51  * b_10;
      c_temp +=  a_52  * b_20;
      c_temp +=  a_53  * b_30;
      c_temp +=  a_54  * b_40;
      c_temp +=  a_55  * b_50;
      c[n + N*5 ] = c_temp;
      c_temp =  a_50  * b_01;
      c_temp +=  a_51  * b_11;
      c_temp +=  a_52  * b_21;
      c_temp +=  a_53  * b_31;
      c_temp +=  a_54  * b_41;
      c_temp +=  a_55  * b_51;
      c[n + N*11 ] = c_temp;
      c_temp =  a_50  * b_02;
      c_temp +=  a_51  * b_12;
      c_temp +=  a_52  * b_22;
      c_temp +=  a_53  * b_32;
      c_temp +=  a_54  * b_42;
      c_temp +=  a_55  * b_52;
      c[n + N*17 ] = c_temp;
      c_temp =  a_50  * b_03;
      c_temp +=  a_51  * b_13;
      c_temp +=  a_52  * b_23;
      c_temp +=  a_53  * b_33;
      c_temp +=  a_54  * b_43;
      c_temp +=  a_55  * b_53;
      c[n + N*23 ] = c_temp;
      c_temp =  a_50  * b_04;
      c_temp +=  a_51  * b_14;
      c_temp +=  a_52  * b_24;
      c_temp +=  a_53  * b_34;
      c_temp +=  a_54  * b_44;
      c_temp +=  a_55  * b_54;
      c[n + N*29 ] = c_temp;
      c_temp =  a_50  * b_05;
      c_temp +=  a_51  * b_15;
      c_temp +=  a_52  * b_25;
      c_temp +=  a_53  * b_35;
      c_temp +=  a_54  * b_45;
      c_temp +=  a_55  * b_55;
      c[n + N*35 ] = c_temp;

    }//n = threadIdx.x + blockIdx.x * blockDim.x
  }//oLoop< nN; ++oLoop){
}

__global__ void raw_reg_c_mult_loop_unroll_const_kn(const float* RESTRICT const a, const float* RESTRICT const b, 
    float* c, const int N)
{
  constexpr int NN = 7168;
  int nN = 1000;
  for (int oLoop = 0; oLoop< nN; ++oLoop){
    for (int n = threadIdx.x + blockIdx.x * blockDim.x;
         n < N;
         n += blockDim.x * gridDim.x) {
      float c_temp;
      float a_00 = a[n + NN*0];
      float b_00 = b[n + NN*0];
      c_temp =  a_00  * b_00;
      float a_01 = a[n + NN*6];
      float b_01 = b[n + NN*6];
      float a_10 = a[n + NN*1];
      float b_10 = b[n + NN*1];
      c_temp +=  a_01  * b_10;
      float a_02 = a[n + NN*12];
      float b_02 = b[n + NN*12];
      float a_20 = a[n + NN*2];
      float b_20 = b[n + NN*2];
      c_temp +=  a_02  * b_20;
      float a_03 = a[n + NN*18];
      float b_03 = b[n + NN*18];
      float a_30 = a[n + NN*3];
      float b_30 = b[n + NN*3];
      c_temp +=  a_03  * b_30;
      float a_04 = a[n + NN*24];
      float b_04 = b[n + NN*24];
      float a_40 = a[n + NN*4];
      float b_40 = b[n + NN*4];
      c_temp +=  a_04  * b_40;
      float a_05 = a[n + NN*30];
      float b_05 = b[n + NN*30];
      float a_50 = a[n + NN*5];
      float b_50 = b[n + NN*5];
      c_temp +=  a_05  * b_50;
      c[n + NN*0 ] = c_temp;
      c_temp =  a_00  * b_01;
      float a_11 = a[n + NN*7];
      float b_11 = b[n + NN*7];
      c_temp +=  a_01  * b_11;
      float a_21 = a[n + NN*8];
      float b_21 = b[n + NN*8];
      c_temp +=  a_02  * b_21;
      float a_31 = a[n + NN*9];
      float b_31 = b[n + NN*9];
      c_temp +=  a_03  * b_31;
      float a_41 = a[n + NN*10];
      float b_41 = b[n + NN*10];
      c_temp +=  a_04  * b_41;
      float a_51 = a[n + NN*11];
      float b_51 = b[n + NN*11];
      c_temp +=  a_05  * b_51;
      c[n + NN*6 ] = c_temp;
      c_temp =  a_00  * b_02;
      float a_12 = a[n + NN*13];
      float b_12 = b[n + NN*13];
      c_temp +=  a_01  * b_12;
      float a_22 = a[n + NN*14];
      float b_22 = b[n + NN*14];
      c_temp +=  a_02  * b_22;
      float a_32 = a[n + NN*15];
      float b_32 = b[n + NN*15];
      c_temp +=  a_03  * b_32;
      float a_42 = a[n + NN*16];
      float b_42 = b[n + NN*16];
      c_temp +=  a_04  * b_42;
      float a_52 = a[n + NN*17];
      float b_52 = b[n + NN*17];
      c_temp +=  a_05  * b_52;
      c[n + NN*12 ] = c_temp;
      c_temp =  a_00  * b_03;
      float a_13 = a[n + NN*19];
      float b_13 = b[n + NN*19];
      c_temp +=  a_01  * b_13;
      float a_23 = a[n + NN*20];
      float b_23 = b[n + NN*20];
      c_temp +=  a_02  * b_23;
      float a_33 = a[n + NN*21];
      float b_33 = b[n + NN*21];
      c_temp +=  a_03  * b_33;
      float a_43 = a[n + NN*22];
      float b_43 = b[n + NN*22];
      c_temp +=  a_04  * b_43;
      float a_53 = a[n + NN*23];
      float b_53 = b[n + NN*23];
      c_temp +=  a_05  * b_53;
      c[n + NN*18 ] = c_temp;
      c_temp =  a_00  * b_04;
      float a_14 = a[n + NN*25];
      float b_14 = b[n + NN*25];
      c_temp +=  a_01  * b_14;
      float a_24 = a[n + NN*26];
      float b_24 = b[n + NN*26];
      c_temp +=  a_02  * b_24;
      float a_34 = a[n + NN*27];
      float b_34 = b[n + NN*27];
      c_temp +=  a_03  * b_34;
      float a_44 = a[n + NN*28];
      float b_44 = b[n + NN*28];
      c_temp +=  a_04  * b_44;
      float a_54 = a[n + NN*29];
      float b_54 = b[n + NN*29];
      c_temp +=  a_05  * b_54;
      c[n + NN*24 ] = c_temp;
      c_temp =  a_00  * b_05;
      float a_15 = a[n + NN*31];
      float b_15 = b[n + NN*31];
      c_temp +=  a_01  * b_15;
      float a_25 = a[n + NN*32];
      float b_25 = b[n + NN*32];
      c_temp +=  a_02  * b_25;
      float a_35 = a[n + NN*33];
      float b_35 = b[n + NN*33];
      c_temp +=  a_03  * b_35;
      float a_45 = a[n + NN*34];
      float b_45 = b[n + NN*34];
      c_temp +=  a_04  * b_45;
      float a_55 = a[n + NN*35];
      float b_55 = b[n + NN*35];
      c_temp +=  a_05  * b_55;
      c[n + NN*30 ] = c_temp;
      c_temp =  a_10  * b_00;
      c_temp +=  a_11  * b_10;
      c_temp +=  a_12  * b_20;
      c_temp +=  a_13  * b_30;
      c_temp +=  a_14  * b_40;
      c_temp +=  a_15  * b_50;
      c[n + NN*1 ] = c_temp;
      c_temp =  a_10  * b_01;
      c_temp +=  a_11  * b_11;
      c_temp +=  a_12  * b_21;
      c_temp +=  a_13  * b_31;
      c_temp +=  a_14  * b_41;
      c_temp +=  a_15  * b_51;
      c[n + NN*7 ] = c_temp;
      c_temp =  a_10  * b_02;
      c_temp +=  a_11  * b_12;
      c_temp +=  a_12  * b_22;
      c_temp +=  a_13  * b_32;
      c_temp +=  a_14  * b_42;
      c_temp +=  a_15  * b_52;
      c[n + NN*13 ] = c_temp;
      c_temp =  a_10  * b_03;
      c_temp +=  a_11  * b_13;
      c_temp +=  a_12  * b_23;
      c_temp +=  a_13  * b_33;
      c_temp +=  a_14  * b_43;
      c_temp +=  a_15  * b_53;
      c[n + NN*19 ] = c_temp;
      c_temp =  a_10  * b_04;
      c_temp +=  a_11  * b_14;
      c_temp +=  a_12  * b_24;
      c_temp +=  a_13  * b_34;
      c_temp +=  a_14  * b_44;
      c_temp +=  a_15  * b_54;
      c[n + NN*25 ] = c_temp;
      c_temp =  a_10  * b_05;
      c_temp +=  a_11  * b_15;
      c_temp +=  a_12  * b_25;
      c_temp +=  a_13  * b_35;
      c_temp +=  a_14  * b_45;
      c_temp +=  a_15  * b_55;
      c[n + NN*31 ] = c_temp;
      c_temp =  a_20  * b_00;
      c_temp +=  a_21  * b_10;
      c_temp +=  a_22  * b_20;
      c_temp +=  a_23  * b_30;
      c_temp +=  a_24  * b_40;
      c_temp +=  a_25  * b_50;
      c[n + NN*2 ] = c_temp;
      c_temp =  a_20  * b_01;
      c_temp +=  a_21  * b_11;
      c_temp +=  a_22  * b_21;
      c_temp +=  a_23  * b_31;
      c_temp +=  a_24  * b_41;
      c_temp +=  a_25  * b_51;
      c[n + NN*8 ] = c_temp;
      c_temp =  a_20  * b_02;
      c_temp +=  a_21  * b_12;
      c_temp +=  a_22  * b_22;
      c_temp +=  a_23  * b_32;
      c_temp +=  a_24  * b_42;
      c_temp +=  a_25  * b_52;
      c[n + NN*14 ] = c_temp;
      c_temp =  a_20  * b_03;
      c_temp +=  a_21  * b_13;
      c_temp +=  a_22  * b_23;
      c_temp +=  a_23  * b_33;
      c_temp +=  a_24  * b_43;
      c_temp +=  a_25  * b_53;
      c[n + NN*20 ] = c_temp;
      c_temp =  a_20  * b_04;
      c_temp +=  a_21  * b_14;
      c_temp +=  a_22  * b_24;
      c_temp +=  a_23  * b_34;
      c_temp +=  a_24  * b_44;
      c_temp +=  a_25  * b_54;
      c[n + NN*26 ] = c_temp;
      c_temp =  a_20  * b_05;
      c_temp +=  a_21  * b_15;
      c_temp +=  a_22  * b_25;
      c_temp +=  a_23  * b_35;
      c_temp +=  a_24  * b_45;
      c_temp +=  a_25  * b_55;
      c[n + NN*32 ] = c_temp;
      c_temp =  a_30  * b_00;
      c_temp +=  a_31  * b_10;
      c_temp +=  a_32  * b_20;
      c_temp +=  a_33  * b_30;
      c_temp +=  a_34  * b_40;
      c_temp +=  a_35  * b_50;
      c[n + NN*3 ] = c_temp;
      c_temp =  a_30  * b_01;
      c_temp +=  a_31  * b_11;
      c_temp +=  a_32  * b_21;
      c_temp +=  a_33  * b_31;
      c_temp +=  a_34  * b_41;
      c_temp +=  a_35  * b_51;
      c[n + NN*9 ] = c_temp;
      c_temp =  a_30  * b_02;
      c_temp +=  a_31  * b_12;
      c_temp +=  a_32  * b_22;
      c_temp +=  a_33  * b_32;
      c_temp +=  a_34  * b_42;
      c_temp +=  a_35  * b_52;
      c[n + NN*15 ] = c_temp;
      c_temp =  a_30  * b_03;
      c_temp +=  a_31  * b_13;
      c_temp +=  a_32  * b_23;
      c_temp +=  a_33  * b_33;
      c_temp +=  a_34  * b_43;
      c_temp +=  a_35  * b_53;
      c[n + NN*21 ] = c_temp;
      c_temp =  a_30  * b_04;
      c_temp +=  a_31  * b_14;
      c_temp +=  a_32  * b_24;
      c_temp +=  a_33  * b_34;
      c_temp +=  a_34  * b_44;
      c_temp +=  a_35  * b_54;
      c[n + NN*27 ] = c_temp;
      c_temp =  a_30  * b_05;
      c_temp +=  a_31  * b_15;
      c_temp +=  a_32  * b_25;
      c_temp +=  a_33  * b_35;
      c_temp +=  a_34  * b_45;
      c_temp +=  a_35  * b_55;
      c[n + NN*33 ] = c_temp;
      c_temp =  a_40  * b_00;
      c_temp +=  a_41  * b_10;
      c_temp +=  a_42  * b_20;
      c_temp +=  a_43  * b_30;
      c_temp +=  a_44  * b_40;
      c_temp +=  a_45  * b_50;
      c[n + NN*4 ] = c_temp;
      c_temp =  a_40  * b_01;
      c_temp +=  a_41  * b_11;
      c_temp +=  a_42  * b_21;
      c_temp +=  a_43  * b_31;
      c_temp +=  a_44  * b_41;
      c_temp +=  a_45  * b_51;
      c[n + NN*10 ] = c_temp;
      c_temp =  a_40  * b_02;
      c_temp +=  a_41  * b_12;
      c_temp +=  a_42  * b_22;
      c_temp +=  a_43  * b_32;
      c_temp +=  a_44  * b_42;
      c_temp +=  a_45  * b_52;
      c[n + NN*16 ] = c_temp;
      c_temp =  a_40  * b_03;
      c_temp +=  a_41  * b_13;
      c_temp +=  a_42  * b_23;
      c_temp +=  a_43  * b_33;
      c_temp +=  a_44  * b_43;
      c_temp +=  a_45  * b_53;
      c[n + NN*22 ] = c_temp;
      c_temp =  a_40  * b_04;
      c_temp +=  a_41  * b_14;
      c_temp +=  a_42  * b_24;
      c_temp +=  a_43  * b_34;
      c_temp +=  a_44  * b_44;
      c_temp +=  a_45  * b_54;
      c[n + NN*28 ] = c_temp;
      c_temp =  a_40  * b_05;
      c_temp +=  a_41  * b_15;
      c_temp +=  a_42  * b_25;
      c_temp +=  a_43  * b_35;
      c_temp +=  a_44  * b_45;
      c_temp +=  a_45  * b_55;
      c[n + NN*34 ] = c_temp;
      c_temp =  a_50  * b_00;
      c_temp +=  a_51  * b_10;
      c_temp +=  a_52  * b_20;
      c_temp +=  a_53  * b_30;
      c_temp +=  a_54  * b_40;
      c_temp +=  a_55  * b_50;
      c[n + NN*5 ] = c_temp;
      c_temp =  a_50  * b_01;
      c_temp +=  a_51  * b_11;
      c_temp +=  a_52  * b_21;
      c_temp +=  a_53  * b_31;
      c_temp +=  a_54  * b_41;
      c_temp +=  a_55  * b_51;
      c[n + NN*11 ] = c_temp;
      c_temp =  a_50  * b_02;
      c_temp +=  a_51  * b_12;
      c_temp +=  a_52  * b_22;
      c_temp +=  a_53  * b_32;
      c_temp +=  a_54  * b_42;
      c_temp +=  a_55  * b_52;
      c[n + NN*17 ] = c_temp;
      c_temp =  a_50  * b_03;
      c_temp +=  a_51  * b_13;
      c_temp +=  a_52  * b_23;
      c_temp +=  a_53  * b_33;
      c_temp +=  a_54  * b_43;
      c_temp +=  a_55  * b_53;
      c[n + NN*23 ] = c_temp;
      c_temp =  a_50  * b_04;
      c_temp +=  a_51  * b_14;
      c_temp +=  a_52  * b_24;
      c_temp +=  a_53  * b_34;
      c_temp +=  a_54  * b_44;
      c_temp +=  a_55  * b_54;
      c[n + NN*29 ] = c_temp;
      c_temp =  a_50  * b_05;
      c_temp +=  a_51  * b_15;
      c_temp +=  a_52  * b_25;
      c_temp +=  a_53  * b_35;
      c_temp +=  a_54  * b_45;
      c_temp +=  a_55  * b_55;
      c[n + NN*35 ] = c_temp;

    }//n = threadIdx.x + blockIdx.x * blockDim.x
  }//oLoop< nN; ++oLoop){
}


/** similar to raw_reg_c_mult_loop_unroll_const_kn but with different grouping of 
    loads and multiplication: all loads needed for the product of a row/column
    are loaded entirely and then all the multiplications/additions happen.

    The code only values from the memory to local variables (i.e. registers)
    if it has not done so for a previous product.
*/
__global__ void raw_reg_c_mult_loop_unroll2_const_kn(const float* RESTRICT const a, const float* RESTRICT const b, 
    float* c, const int N)
{
  constexpr int nN = 1000;
  for (int oLoop = 0; oLoop< nN; ++oLoop){
    for (int n = threadIdx.x + blockIdx.x * blockDim.x;
         n < N;
         n += blockDim.x * gridDim.x) {

	 float c_temp;
	 float a_00 = a[n + N*0];
	 float b_00 = b[n + N*0];
	 float a_01 = a[n + N*6];
	 float b_01 = b[n + N*6];
	 float a_10 = a[n + N*1];
	 float b_10 = b[n + N*1];
	 float a_02 = a[n + N*12];
	 float b_02 = b[n + N*12];
	 float a_20 = a[n + N*2];
	 float b_20 = b[n + N*2];
	 float a_03 = a[n + N*18];
	 float b_03 = b[n + N*18];
	 float a_30 = a[n + N*3];
	 float b_30 = b[n + N*3];
	 float a_04 = a[n + N*24];
	 float b_04 = b[n + N*24];
	 float a_40 = a[n + N*4];
	 float b_40 = b[n + N*4];
	 float a_05 = a[n + N*30];
	 float b_05 = b[n + N*30];
	 float a_50 = a[n + N*5];
	 float b_50 = b[n + N*5];
	 c_temp =  a_00  * b_00;
	 c_temp +=  a_01  * b_10;
	 c_temp +=  a_02  * b_20;
	 c_temp +=  a_03  * b_30;
	 c_temp +=  a_04  * b_40;
	 c_temp +=  a_05  * b_50;
	 c[n + N*0 ] = c_temp;
	 float a_11 = a[n + N*7];
	 float b_11 = b[n + N*7];
	 float a_21 = a[n + N*8];
	 float b_21 = b[n + N*8];
	 float a_31 = a[n + N*9];
	 float b_31 = b[n + N*9];
	 float a_41 = a[n + N*10];
	 float b_41 = b[n + N*10];
	 float a_51 = a[n + N*11];
	 float b_51 = b[n + N*11];
	 c_temp =  a_00  * b_01;
	 c_temp +=  a_01  * b_11;
	 c_temp +=  a_02  * b_21;
	 c_temp +=  a_03  * b_31;
	 c_temp +=  a_04  * b_41;
	 c_temp +=  a_05  * b_51;
	 c[n + N*6 ] = c_temp;
	 float a_12 = a[n + N*13];
	 float b_12 = b[n + N*13];
	 float a_22 = a[n + N*14];
	 float b_22 = b[n + N*14];
	 float a_32 = a[n + N*15];
	 float b_32 = b[n + N*15];
	 float a_42 = a[n + N*16];
	 float b_42 = b[n + N*16];
	 float a_52 = a[n + N*17];
	 float b_52 = b[n + N*17];
	 c_temp =  a_00  * b_02;
	 c_temp +=  a_01  * b_12;
	 c_temp +=  a_02  * b_22;
	 c_temp +=  a_03  * b_32;
	 c_temp +=  a_04  * b_42;
	 c_temp +=  a_05  * b_52;
	 c[n + N*12 ] = c_temp;
	 float a_13 = a[n + N*19];
	 float b_13 = b[n + N*19];
	 float a_23 = a[n + N*20];
	 float b_23 = b[n + N*20];
	 float a_33 = a[n + N*21];
	 float b_33 = b[n + N*21];
	 float a_43 = a[n + N*22];
	 float b_43 = b[n + N*22];
	 float a_53 = a[n + N*23];
	 float b_53 = b[n + N*23];
	 c_temp =  a_00  * b_03;
	 c_temp +=  a_01  * b_13;
	 c_temp +=  a_02  * b_23;
	 c_temp +=  a_03  * b_33;
	 c_temp +=  a_04  * b_43;
	 c_temp +=  a_05  * b_53;
	 c[n + N*18 ] = c_temp;
	 float a_14 = a[n + N*25];
	 float b_14 = b[n + N*25];
	 float a_24 = a[n + N*26];
	 float b_24 = b[n + N*26];
	 float a_34 = a[n + N*27];
	 float b_34 = b[n + N*27];
	 float a_44 = a[n + N*28];
	 float b_44 = b[n + N*28];
	 float a_54 = a[n + N*29];
	 float b_54 = b[n + N*29];
	 c_temp =  a_00  * b_04;
	 c_temp +=  a_01  * b_14;
	 c_temp +=  a_02  * b_24;
	 c_temp +=  a_03  * b_34;
	 c_temp +=  a_04  * b_44;
	 c_temp +=  a_05  * b_54;
	 c[n + N*24 ] = c_temp;
	 float a_15 = a[n + N*31];
	 float b_15 = b[n + N*31];
	 float a_25 = a[n + N*32];
	 float b_25 = b[n + N*32];
	 float a_35 = a[n + N*33];
	 float b_35 = b[n + N*33];
	 float a_45 = a[n + N*34];
	 float b_45 = b[n + N*34];
	 float a_55 = a[n + N*35];
	 float b_55 = b[n + N*35];
	 c_temp =  a_00  * b_05;
	 c_temp +=  a_01  * b_15;
	 c_temp +=  a_02  * b_25;
	 c_temp +=  a_03  * b_35;
	 c_temp +=  a_04  * b_45;
	 c_temp +=  a_05  * b_55;
	 c[n + N*30 ] = c_temp;
	 c_temp =  a_10  * b_00;
	 c_temp +=  a_11  * b_10;
	 c_temp +=  a_12  * b_20;
	 c_temp +=  a_13  * b_30;
	 c_temp +=  a_14  * b_40;
	 c_temp +=  a_15  * b_50;
	 c[n + N*1 ] = c_temp;
	 c_temp =  a_10  * b_01;
	 c_temp +=  a_11  * b_11;
	 c_temp +=  a_12  * b_21;
	 c_temp +=  a_13  * b_31;
	 c_temp +=  a_14  * b_41;
	 c_temp +=  a_15  * b_51;
	 c[n + N*7 ] = c_temp;
	 c_temp =  a_10  * b_02;
	 c_temp +=  a_11  * b_12;
	 c_temp +=  a_12  * b_22;
	 c_temp +=  a_13  * b_32;
	 c_temp +=  a_14  * b_42;
	 c_temp +=  a_15  * b_52;
	 c[n + N*13 ] = c_temp;
	 c_temp =  a_10  * b_03;
	 c_temp +=  a_11  * b_13;
	 c_temp +=  a_12  * b_23;
	 c_temp +=  a_13  * b_33;
	 c_temp +=  a_14  * b_43;
	 c_temp +=  a_15  * b_53;
	 c[n + N*19 ] = c_temp;
	 c_temp =  a_10  * b_04;
	 c_temp +=  a_11  * b_14;
	 c_temp +=  a_12  * b_24;
	 c_temp +=  a_13  * b_34;
	 c_temp +=  a_14  * b_44;
	 c_temp +=  a_15  * b_54;
	 c[n + N*25 ] = c_temp;
	 c_temp =  a_10  * b_05;
	 c_temp +=  a_11  * b_15;
	 c_temp +=  a_12  * b_25;
	 c_temp +=  a_13  * b_35;
	 c_temp +=  a_14  * b_45;
	 c_temp +=  a_15  * b_55;
	 c[n + N*31 ] = c_temp;
	 c_temp =  a_20  * b_00;
	 c_temp +=  a_21  * b_10;
	 c_temp +=  a_22  * b_20;
	 c_temp +=  a_23  * b_30;
	 c_temp +=  a_24  * b_40;
	 c_temp +=  a_25  * b_50;
	 c[n + N*2 ] = c_temp;
	 c_temp =  a_20  * b_01;
	 c_temp +=  a_21  * b_11;
	 c_temp +=  a_22  * b_21;
	 c_temp +=  a_23  * b_31;
	 c_temp +=  a_24  * b_41;
	 c_temp +=  a_25  * b_51;
	 c[n + N*8 ] = c_temp;
	 c_temp =  a_20  * b_02;
	 c_temp +=  a_21  * b_12;
	 c_temp +=  a_22  * b_22;
	 c_temp +=  a_23  * b_32;
	 c_temp +=  a_24  * b_42;
	 c_temp +=  a_25  * b_52;
	 c[n + N*14 ] = c_temp;
	 c_temp =  a_20  * b_03;
	 c_temp +=  a_21  * b_13;
	 c_temp +=  a_22  * b_23;
	 c_temp +=  a_23  * b_33;
	 c_temp +=  a_24  * b_43;
	 c_temp +=  a_25  * b_53;
	 c[n + N*20 ] = c_temp;
	 c_temp =  a_20  * b_04;
	 c_temp +=  a_21  * b_14;
	 c_temp +=  a_22  * b_24;
	 c_temp +=  a_23  * b_34;
	 c_temp +=  a_24  * b_44;
	 c_temp +=  a_25  * b_54;
	 c[n + N*26 ] = c_temp;
	 c_temp =  a_20  * b_05;
	 c_temp +=  a_21  * b_15;
	 c_temp +=  a_22  * b_25;
	 c_temp +=  a_23  * b_35;
	 c_temp +=  a_24  * b_45;
	 c_temp +=  a_25  * b_55;
	 c[n + N*32 ] = c_temp;
	 c_temp =  a_30  * b_00;
	 c_temp +=  a_31  * b_10;
	 c_temp +=  a_32  * b_20;
	 c_temp +=  a_33  * b_30;
	 c_temp +=  a_34  * b_40;
	 c_temp +=  a_35  * b_50;
	 c[n + N*3 ] = c_temp;
	 c_temp =  a_30  * b_01;
	 c_temp +=  a_31  * b_11;
	 c_temp +=  a_32  * b_21;
	 c_temp +=  a_33  * b_31;
	 c_temp +=  a_34  * b_41;
	 c_temp +=  a_35  * b_51;
	 c[n + N*9 ] = c_temp;
	 c_temp =  a_30  * b_02;
	 c_temp +=  a_31  * b_12;
	 c_temp +=  a_32  * b_22;
	 c_temp +=  a_33  * b_32;
	 c_temp +=  a_34  * b_42;
	 c_temp +=  a_35  * b_52;
	 c[n + N*15 ] = c_temp;
	 c_temp =  a_30  * b_03;
	 c_temp +=  a_31  * b_13;
	 c_temp +=  a_32  * b_23;
	 c_temp +=  a_33  * b_33;
	 c_temp +=  a_34  * b_43;
	 c_temp +=  a_35  * b_53;
	 c[n + N*21 ] = c_temp;
	 c_temp =  a_30  * b_04;
	 c_temp +=  a_31  * b_14;
	 c_temp +=  a_32  * b_24;
	 c_temp +=  a_33  * b_34;
	 c_temp +=  a_34  * b_44;
	 c_temp +=  a_35  * b_54;
	 c[n + N*27 ] = c_temp;
	 c_temp =  a_30  * b_05;
	 c_temp +=  a_31  * b_15;
	 c_temp +=  a_32  * b_25;
	 c_temp +=  a_33  * b_35;
	 c_temp +=  a_34  * b_45;
	 c_temp +=  a_35  * b_55;
	 c[n + N*33 ] = c_temp;
	 c_temp =  a_40  * b_00;
	 c_temp +=  a_41  * b_10;
	 c_temp +=  a_42  * b_20;
	 c_temp +=  a_43  * b_30;
	 c_temp +=  a_44  * b_40;
	 c_temp +=  a_45  * b_50;
	 c[n + N*4 ] = c_temp;
	 c_temp =  a_40  * b_01;
	 c_temp +=  a_41  * b_11;
	 c_temp +=  a_42  * b_21;
	 c_temp +=  a_43  * b_31;
	 c_temp +=  a_44  * b_41;
	 c_temp +=  a_45  * b_51;
	 c[n + N*10 ] = c_temp;
	 c_temp =  a_40  * b_02;
	 c_temp +=  a_41  * b_12;
	 c_temp +=  a_42  * b_22;
	 c_temp +=  a_43  * b_32;
	 c_temp +=  a_44  * b_42;
	 c_temp +=  a_45  * b_52;
	 c[n + N*16 ] = c_temp;
	 c_temp =  a_40  * b_03;
	 c_temp +=  a_41  * b_13;
	 c_temp +=  a_42  * b_23;
	 c_temp +=  a_43  * b_33;
	 c_temp +=  a_44  * b_43;
	 c_temp +=  a_45  * b_53;
	 c[n + N*22 ] = c_temp;
	 c_temp =  a_40  * b_04;
	 c_temp +=  a_41  * b_14;
	 c_temp +=  a_42  * b_24;
	 c_temp +=  a_43  * b_34;
	 c_temp +=  a_44  * b_44;
	 c_temp +=  a_45  * b_54;
	 c[n + N*28 ] = c_temp;
	 c_temp =  a_40  * b_05;
	 c_temp +=  a_41  * b_15;
	 c_temp +=  a_42  * b_25;
	 c_temp +=  a_43  * b_35;
	 c_temp +=  a_44  * b_45;
	 c_temp +=  a_45  * b_55;
	 c[n + N*34 ] = c_temp;
	 c_temp =  a_50  * b_00;
	 c_temp +=  a_51  * b_10;
	 c_temp +=  a_52  * b_20;
	 c_temp +=  a_53  * b_30;
	 c_temp +=  a_54  * b_40;
	 c_temp +=  a_55  * b_50;
	 c[n + N*5 ] = c_temp;
	 c_temp =  a_50  * b_01;
	 c_temp +=  a_51  * b_11;
	 c_temp +=  a_52  * b_21;
	 c_temp +=  a_53  * b_31;
	 c_temp +=  a_54  * b_41;
	 c_temp +=  a_55  * b_51;
	 c[n + N*11 ] = c_temp;
	 c_temp =  a_50  * b_02;
	 c_temp +=  a_51  * b_12;
	 c_temp +=  a_52  * b_22;
	 c_temp +=  a_53  * b_32;
	 c_temp +=  a_54  * b_42;
	 c_temp +=  a_55  * b_52;
	 c[n + N*17 ] = c_temp;
	 c_temp =  a_50  * b_03;
	 c_temp +=  a_51  * b_13;
	 c_temp +=  a_52  * b_23;
	 c_temp +=  a_53  * b_33;
	 c_temp +=  a_54  * b_43;
	 c_temp +=  a_55  * b_53;
	 c[n + N*23 ] = c_temp;
	 c_temp =  a_50  * b_04;
	 c_temp +=  a_51  * b_14;
	 c_temp +=  a_52  * b_24;
	 c_temp +=  a_53  * b_34;
	 c_temp +=  a_54  * b_44;
	 c_temp +=  a_55  * b_54;
	 c[n + N*29 ] = c_temp;
	 c_temp =  a_50  * b_05;
	 c_temp +=  a_51  * b_15;
	 c_temp +=  a_52  * b_25;
	 c_temp +=  a_53  * b_35;
	 c_temp +=  a_54  * b_45;
	 c_temp +=  a_55  * b_55;
	 c[n + N*35 ] = c_temp;
    }//n = threadIdx.x + blockIdx.x * blockDim.x
  }//oLoop< nN; ++oLoop){
}

__global__ void raw_shared_mult_kn(const float* RESTRICT a, const float* RESTRICT b, float* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
       n < N;
       n += blockDim.x * gridDim.x) {

    int tix = threadIdx.x;

    __shared__ float sh_a[36][block_size];
    __shared__ float sh_b[36][block_size];

    for (int i = 0; i < 36; ++i) {
      sh_a[i][tix] = a[n + N*i];
    }
    for (int i = 0; i < 36; ++i) {
      sh_b[i][tix] = b[n + N*i];
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

__global__ void raw_reg_mult_kn(const float* RESTRICT a, const float* RESTRICT b, float* c, const int N)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x;
      n < N;
      n += blockDim.x * gridDim.x) {

    float reg_a[36];
    float reg_b[36];

    for (int i = 0; i < 36; ++i) {
      reg_a[i] = a[n + N*i];
    }
    for (int i = 0; i < 36; ++i) {
      reg_b[i] = b[n + N*i];
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


void fill(int N, float* a, float* b, float* c, float *h)
{
  int n = 36*N;
  for (int i = 0; i < n; ++i) {
    h[i] = 1.0;
  }
  cudaMemcpy(a, h, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b, h, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(c, 0, n*sizeof(float));
  cudaCheckErrorSync();
}

bool check(int N, float* c, float *h)
{
  const float eps = 1e-30;
  int n = 36*N;
  cudaMemcpy(h, c, n*sizeof(float), cudaMemcpyDeviceToHost);
  return (std::abs(h[0] - h[36]) < eps) && (std::abs(h[0] - 6.0f) < eps);
}

void raw_run_naive_mul(int N, int iter)
{
  /*constexpr int N = 3200000;//10000000;*/
  /*constexpr int N = 100000;*/
  float* a;
  float* b;
  float* c;
  float h[36*N];

  cudaMalloc((void**)&a, 36*N*sizeof(float));
  cudaMalloc((void**)&b, 36*N*sizeof(float));
  cudaMalloc((void**)&c, 36*N*sizeof(float));
  cudaCheckErrorSync();

  dim3 grid (((N-1)/block_size + 1), 1, 1);
  dim3 block (block_size, 1, 1);

  fill(N, a, b, c, h);

  for (int i = 0; i < iter; ++i)
    raw_naive_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, h));

  for (int i = 0; i < iter; ++i)
    raw_reg_c_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, h));

  for (int i = 0; i < iter; ++i)
    raw_shared_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, h));

  for (int i = 0; i < iter; ++i)
    raw_reg_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();

  raw_regall_c_mult_loop_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  //no checks here because the operations are not the same

  raw_reg_c_mult_loop_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, h));

  raw_reg_c_mult_loop_kn_vl <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, h));

  raw_reg_c_mult_loop_kn_vl_vs <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, h));

  raw_reg_c_mult_loop_kn_vl_transp <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  // since this kernel assumes a transposed layout
  // the check may fail in the future
  assert(check(N, c, h));

  raw_reg_c_mult_loop_unroll_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, h));

  raw_reg_c_mult_loop_unroll_const_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, h));

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaCheckErrorSync();
}

#ifdef EIGEN_TEST
#include <Eigen/Dense>

using Matrix66 = Eigen::Matrix<float, 6, 6, Eigen::AutoAlign>;

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

void eigen_run_naive_mul(int N, int iter)
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

  for (int i = 0; i < iter; ++i)
    eigen_reg_c_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();

  for (int i = 0; i < iter; ++i)
    eigen_reg_mult_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();

  raw_reg_c_mult_loop_unroll2_const_kn <<< grid, block >>> (a, b, c, N);
  cudaCheckErrorSync();
  assert(check(N, c, h));

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaCheckErrorSync();
}
#endif

