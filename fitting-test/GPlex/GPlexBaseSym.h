#ifndef GPlexBaseSym_H
#define GPlexBaseSym_H

#include "GPlexBaseCommon.h"
#include "GPlexBase.h"

//==============================================================================
// GPlexBaseSym
//==============================================================================

namespace GPlexBase
{
template<typename T, idx_t D, idx_t N>
class GPlexBaseSym
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
      /// size of the whole GPlexBase
      kTotSize = N * kSize
   };

   T fArray[kTotSize] __attribute__((aligned(64)));
};
}
#endif
