#ifndef MatriplexSym_H
#define MatriplexSym_H

#include "MatriplexCommon.h"
#include "Matriplex.h"

//==============================================================================
// MatriplexSym
//==============================================================================

namespace Matriplex
{
template<typename T, idx_t D, idx_t N>
class MatriplexSym
{
public:
   typedef T value_type;

   enum
   {
      /// no. of matrix rows
      kRows = D,
      /// no. of matrix columns
      kCols = D,
      /// no of elements: lower triangle
      kSize = (D + 1) * D / 2,
      /// size of the whole matriplex
      kTotSize = N * kSize
   };

   T fArray[kTotSize] __attribute__((aligned(64)));
};

template<typename T, idx_t D, idx_t N> using MPlexSym = MatriplexSym<T, D, N>;

}

#endif
