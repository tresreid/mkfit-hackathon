#pragma once
//#define EIGEN_TEST

void run_naive_mul(int N, int iter, bool pauseProf);
void raw_run_naive_mul(int N, int iter, bool pauseProf);
void eigen_run_naive_mul(int N, int iter, bool pauseProf);
void propagation_test(int N);
