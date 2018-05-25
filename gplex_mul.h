#pragma once
//#define EIGEN_TEST

void run_naive_mul(int iter);
void raw_run_naive_mul(int iter);
void eigen_run_naive_mul(int iter);
void propagation_test(int N);

constexpr int Nwidth = 7168;
