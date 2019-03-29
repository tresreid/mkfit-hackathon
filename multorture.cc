#include <cstdio>
#include "gplex_mul.h"

int main(int argc, char** argv) {
  {
    // size_t testN = 7168; has moved to gplex_mul.h
    const int testIter = 1000;
    const bool managed = argc > 1;

    printf("running %s\n", managed ? "managed" : "unmanaged");

    printf("Run run_naive_mul\n");
    run_naive_mul(testIter, managed);

    printf("Run raw_run_naive_mul\n");
    raw_run_naive_mul(testIter, managed);

#ifdef EIGEN_TEST
    printf("Run eigen_run_naive_mul\n");
    eigen_run_naive_mul(testIter, managed);
#endif
  }

  printf("Bye\n");
  return 0;
}
