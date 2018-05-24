#include <cstdio>
#include "gplex_mul.h"

int main() {

  {
    size_t testN = 10000;
    int testIter = 1;
    bool pauseProf = false;

    printf("Run run_naive_mul\n");
    run_naive_mul(testN, testIter, pauseProf);

    printf("Run raw_run_naive_mul\n");
    raw_run_naive_mul(testN, testIter, pauseProf);

#ifdef EIGEN_TEST
    printf("Run eigen_run_naive_mul\n");
    eigen_run_naive_mul(testN, testIter, pauseProf);
#endif
  }

  printf("Bye\n");
  return 0;
}
