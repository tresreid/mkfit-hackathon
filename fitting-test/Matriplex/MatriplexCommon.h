#ifndef MatriplexCommon_H
#define MatriplexCommon_H

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "immintrin.h"

namespace Matriplex
{
   typedef int idx_t;

   inline void align_check(const char* pref, void *adr)
   {
      printf("%s 0x%llx  -  modulo 64 = %lld\n", pref, (long long unsigned)adr, (long long)adr%64);
   }
}

#endif
