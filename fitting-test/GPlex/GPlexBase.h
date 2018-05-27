#ifndef GPlexBase_H
#define GPlexBase_H

#include "GPlexBaseCommon.h"

namespace GPlexBase
{

//------------------------------------------------------------------------------

template<typename T, idx_t D1, idx_t D2, idx_t N>
class GPlexBase
{
public:
   typedef T value_type;

   enum
   {
      /// return no. of matrix rows
      kRows = D1,
      /// return no. of matrix columns
      kCols = D2,
      /// return no of elements: rows*columns
      kSize = D1 * D2,
      /// size of the whole GPlexBase
      kTotSize = N * kSize
   };

   T fArray[kTotSize] __attribute__((aligned(64)));
};
}

#endif
