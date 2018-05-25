#include <cstdio>
#include "gplex_mul.h"

int main() {

  {
    // size_t testN = 7168; has moved to gplex_mul.h
    int testIter = 1;

    printf("Run run_naive_mul\n");
    run_naive_mul(testIter);

    printf("Run raw_run_naive_mul\n");
    raw_run_naive_mul(testIter);

#ifdef EIGEN_TEST
    printf("Run eigen_run_naive_mul\n");
    eigen_run_naive_mul(testIter);
#endif
  }

  printf("Bye\n");
  return 0;
}
