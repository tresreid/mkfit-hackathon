#ifndef _matrix_
#define _matrix_

#include "Math/SMatrix.h"
#include "Config.h"

typedef ROOT::Math::SMatrix<float,6,6,ROOT::Math::MatRepSym<float,6> >    SMatrixSym66;
typedef ROOT::Math::SMatrix<float,6> SMatrix66;
typedef ROOT::Math::SVector<float,6> SVector6;

typedef ROOT::Math::SMatrix<float,3> SMatrix33;
typedef ROOT::Math::SMatrix<float,3,3,ROOT::Math::MatRepSym<float,3> >    SMatrixSym33;
typedef ROOT::Math::SVector<float,3> SVector3;

typedef ROOT::Math::SMatrix<float,2> SMatrix22;
typedef ROOT::Math::SMatrix<float,2,2,ROOT::Math::MatRepSym<float,2> >    SMatrixSym22;
typedef ROOT::Math::SVector<float,2> SVector2;

typedef ROOT::Math::SMatrix<float,3,6> SMatrix36;
typedef ROOT::Math::SMatrix<float,6,3> SMatrix63;

typedef ROOT::Math::SMatrix<float,2,6> SMatrix26;
typedef ROOT::Math::SMatrix<float,6,2> SMatrix62;

// should work with any SMatrix
template<typename Matrix>
void dumpMatrix(Matrix m)
{
  for (int r=0;r<m.kRows;++r) {
    for (int c=0;c<m.kCols;++c) {
      std::cout << std::setw(12) << m.At(r,c) << " ";
    }
    std::cout << std::endl;
  }
}


//==============================================================================

// This should go elsewhere, eventually.

#include <sys/time.h>

inline double dtime()
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime,(struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return( tseconds );
}

CUDA_CALLABLE
inline float hipo(float x, float y)
{
  return std::sqrt(x*x + y*y);
}

CUDA_CALLABLE
inline void sincos4(const float x, float& sin, float& cos)
{
   // Had this writen with explicit division by factorial.
   // The *whole* fitting test ran like 2.5% slower on MIC, sigh.

   const float x2 = x*x;
   cos  = 1.f - 0.5f*x2 + 0.04166667f*x2*x2;
   sin  = x - 0.16666667f*x*x2;
}
//==============================================================================

#ifdef __INTEL_COMPILER
  #define ASSUME_ALIGNED(a, b) __assume_aligned(a, b)
#else
  #define ASSUME_ALIGNED(a, b) a = static_cast<decltype(a)>(__builtin_assume_aligned(a, b))
#endif

#include "immintrin.h"
#include "GPlex/GPlexBaseSym.h"

constexpr GPlexBase::idx_t NN =  MPT_SIZE; // "Length" of MPlex.

constexpr GPlexBase::idx_t LL =  6; // Dimension of large/long  MPlex entities
constexpr GPlexBase::idx_t HH =  3; // Dimension of small/short MPlex entities

typedef GPlexBase::GPlexBase<float, LL, LL, NN>   MPlexLL;
typedef GPlexBase::GPlexBase<float, LL,  1, NN>   MPlexLV;
typedef GPlexBase::GPlexBaseSym<float, LL,  NN>   MPlexLS;

typedef GPlexBase::GPlexBase<float, HH, HH, NN>   MPlexHH;
typedef GPlexBase::GPlexBase<float, HH,  1, NN>   MPlexHV;
typedef GPlexBase::GPlexBaseSym<float, HH,  NN>   MPlexHS;

typedef GPlexBase::GPlexBase<float, 2,  2, NN>    MPlex22;
typedef GPlexBase::GPlexBase<float, 2,  1, NN>    MPlex2V;
typedef GPlexBase::GPlexBaseSym<float,  2, NN>    MPlex2S;

typedef GPlexBase::GPlexBase<float, LL, HH, NN>   MPlexLH;
typedef GPlexBase::GPlexBase<float, HH, LL, NN>   MPlexHL;

typedef GPlexBase::GPlexBase<float, LL,  2, NN>   MPlexL2;

typedef GPlexBase::GPlexBase<float, 1, 1, NN>     MPlexQF;
typedef GPlexBase::GPlexBase<int,   1, 1, NN>     MPlexQI;

typedef GPlexBase::GPlexBase<bool,  1, 1, NN>     MPlexQB;


//==============================================================================

#include <random>

extern std::default_random_engine            g_gen;
extern std::normal_distribution<float>       g_gaus;
extern std::uniform_real_distribution<float> g_unif;

#endif
