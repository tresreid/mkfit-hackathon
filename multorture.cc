#include <cstdio>
#include "gplex_mul.h"

int main() {

  {
    size_t testN = 7168;
    int testIter = 1;

    printf("Run run_naive_mul\n");
    run_naive_mul(testN, testIter);

    printf("Run raw_run_naive_mul\n");
    raw_run_naive_mul(testN, testIter);

#ifdef EIGEN_TEST
    printf("Run eigen_run_naive_mul\n");
    eigen_run_naive_mul(testN, testIter);
#endif
  }

  printf("Bye\n");
  return 0;
}
