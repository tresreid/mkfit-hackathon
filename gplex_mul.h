#pragma once
#define EIGEN_TEST

#if defined(__CUDACC__)
#define CUB_STDERR
#include "gpu_utils.h"
#include "cuda_profiler_api.h"

#if 0 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
#define RESTRICT /* */
#else
#define RESTRICT __restrict__
#endif
#endif

#include <iostream>
#include <cassert>

constexpr int block_size = 64;
constexpr int Nwidth = 4096;

void run_naive_mul(int iter, bool managed);
void raw_run_naive_mul(int iter, bool managed);
void eigen_run_naive_mul(int iter, bool managed);
void propagation_test(int N);
