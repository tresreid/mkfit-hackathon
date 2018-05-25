#include "Config.h"
#include "Hit.h"
#include "kalmanUpdater_kernels.h"
#include "computeChi2_kernels.h"
#include "gpu_utils.h"

#include <trove/ptr.h>

// TODO: Clean all the hard-coded #define
#define LS 21
#define HS 6
#define LH 18
#define HV 3

#define BLOCK_SIZE_X 32


__device__ void subtract_matrix(const float *a, const int aN, 
                                const float *b, const int bN, 
                                      float *c, const int cN,
                                const int size, const int n) {
  float ra [3];
  float rb [3];
  for (int i = 0; i < size; i+=3) {
    ra[0] = a[(i+0) * aN + n]; 
    ra[1] = a[(i+1) * aN + n]; 
    ra[2] = a[(i+2) * aN + n]; 

    rb[0] = b[(i+0) * aN + n]; 
    rb[1] = b[(i+1) * aN + n]; 
    rb[2] = b[(i+2) * aN + n]; 

    c[(i+0)*cN + n] = ra[0] - rb[0];
    c[(i+1)*cN + n] = ra[1] - rb[1];
    c[(i+2)*cN + n] = ra[2] - rb[2];
    /*c[i*cN + n] = a[i*aN + n] - b[i*bN + n];*/
  }
}

__device__ float getHypot_fn(const float x, const float y)
{
  return sqrt(x*x + y*y);
}

__device__
void KalmanHTG_fn(const GPlexRegQF& a00, const GPlexRegQF& a01,
                  const GPlexReg2S &b, GPlexRegHH &c)
{

   // HTG  = rot * res_loc
   //   C  =  A  *    B   

   // Based on script generation and adapted to custom sizes.
      c[ 0] = a00[0]*b[ 0];
      c[ 1] = a00[0]*b[ 1];
      c[ 2] = 0.;
      c[ 3] = a01[0]*b[ 0];
      c[ 4] = a01[0]*b[ 1];
      c[ 5] = 0.;
      c[ 6] = b[ 1];
      c[ 7] = b[ 2];
      c[ 8] = 0.;
}

__device__
void KalmanGain_fn(const GPlexLS &A, const GPlexRegHH &b, GPlexRegLH &c, const int n)
{
  // C = A * B, C is 6x3, A is 6x6 sym , B is 6x3
  using T = float;
  float *a = A.ptr;
  int aN = A.stride; int an = n;  // Global array
  int bN = 1;        int bn = 0;  // Register array
  int cN = 1;        int cn = 0;

#include "KalmanGain.ah"
}

#include "KalmanUtilsMPlex.icc"

__device__
void KHMult_fn(const GPlexRegLH &A, 
               const GPlexRegQF& B00,
               const GPlexRegQF& B01,
               GPlexRegLL &C)
{
  KHMult_imp(A, B00, B01, C, 0, 1);
}

__device__
void KHC_fn(const GPlexRegLL &a, const GPlexLS &B, GPlexLS &C, const int n)
{
  // C = A * B, C is 6x6, A is 6x6 , B is 6x6 sym
  using T = float;
                 int aN = 1; int an = 0;  // Register array
  T *b = B.ptr;  int bN = B.stride;  int bn = n;
  T *c = C.ptr;  int cN = C.stride;  int cn = n;
  /*trove::coalesced_ptr<T> b {B.ptr};  int bN = B.stride;  int bn = n;*/
  /*trove::coalesced_ptr<T> c {C.ptr};  int cN = C.stride;  int cn = n;*/
#if 0
#include "KHC.ah"
#else
      T rb[5];

      rb[0] = b[ 0*bN+bn];
      rb[1] = b[ 1*bN+bn];
      rb[2] = b[ 2*bN+bn];
      rb[3] = b[ 3*bN+bn];
      rb[4] = b[ 4*bN+bn];
      c[ 0*cN+cn] = a[ 0*aN+an]*rb[ 0] + a[ 1*aN+an]*rb[ 1] + a[ 2*aN+an]*rb[ 3];
      c[ 1*cN+cn] = a[ 6*aN+an]*rb[ 0] + a[ 7*aN+an]*rb[ 1] + a[ 8*aN+an]*rb[ 3];
      c[ 2*cN+cn] = a[ 6*aN+an]*rb[ 1] + a[ 7*aN+an]*rb[ 2] + a[ 8*aN+an]*rb[ 4];

      c[ 3*cN+cn] = a[12*aN+an]*rb[ 0] + a[13*aN+an]*rb[ 1] + a[14*aN+an]*rb[ 3];
      c[ 4*cN+cn] = a[12*aN+an]*rb[ 1] + a[13*aN+an]*rb[ 2] + a[14*aN+an]*rb[ 4];
      c[ 5*cN+cn] = a[12*aN+an]*rb[ 3] + a[13*aN+an]*rb[ 4] + a[14*aN+an]*b[ 5*bN+bn];

      c[ 6*cN+cn] = a[18*aN+an]*rb[ 0] + a[19*aN+an]*rb[ 1] + a[20*aN+an]*rb[ 3];
      c[ 7*cN+cn] = a[18*aN+an]*rb[ 1] + a[19*aN+an]*rb[ 2] + a[20*aN+an]*rb[ 4];
      c[ 8*cN+cn] = a[18*aN+an]*rb[ 3] + a[19*aN+an]*rb[ 4] + a[20*aN+an]*b[ 5*bN+bn];

      c[10*cN+cn] = a[24*aN+an]*rb[ 0] + a[25*aN+an]*rb[ 1] + a[26*aN+an]*rb[ 3];
      c[11*cN+cn] = a[24*aN+an]*rb[ 1] + a[25*aN+an]*rb[ 2] + a[26*aN+an]*rb[ 4];
      c[15*cN+cn] = a[30*aN+an]*rb[ 0] + a[31*aN+an]*rb[ 1] + a[32*aN+an]*rb[ 3];
      c[16*cN+cn] = a[30*aN+an]*rb[ 1] + a[31*aN+an]*rb[ 2] + a[32*aN+an]*rb[ 4];

      rb[0] = b[ 5*bN+bn];
      c[12*cN+cn] = a[24*aN+an]*rb[ 3] + a[25*aN+an]*rb[ 4] + a[26*aN+an]*rb[ 0];
      c[17*cN+cn] = a[30*aN+an]*rb[ 3] + a[31*aN+an]*rb[ 4] + a[32*aN+an]*rb[ 0];

      rb[0] = b[ 6*bN+bn];
      rb[1] = b[ 7*bN+bn];
      rb[2] = b[ 8*bN+bn];
      c[ 9*cN+cn] = a[18*aN+an]*rb[ 0] + a[19*aN+an]*rb[ 1] + a[20*aN+an]*rb[ 1];
      c[13*cN+cn] = a[24*aN+an]*rb[ 0] + a[25*aN+an]*rb[ 1] + a[26*aN+an]*rb[ 2];
      c[18*cN+cn] = a[30*aN+an]*rb[ 0] + a[31*aN+an]*rb[ 1] + a[32*aN+an]*rb[ 2];

      rb[0] = b[ 10*bN+bn];
      rb[1] = b[ 11*bN+bn];
      rb[2] = b[ 12*bN+bn];
      c[14*cN+cn] = a[24*aN+an]*rb[ 0] + a[25*aN+an]*rb[1] + a[26*aN+an]*rb[2];
      c[19*cN+cn] = a[30*aN+an]*rb[ 0] + a[31*aN+an]*rb[1] + a[32*aN+an]*rb[2];

      c[20*cN+cn] = a[30*aN+an]*b[15*bN+bn] + a[31*aN+an]*b[16*bN+bn] + a[32*aN+an]*b[17*bN+bn];
#endif
}

__device__
void KHMult_polar_fn(const GPlexRegLH &a, 
               const GPlexRegQF& b00,
               const GPlexRegQF& b01,
               GPlexRegLH &c)
{
      c[0] = a[0]*b00[0];
      c[1] = a[0]*b01[0];
      c[2] = a[1];
      c[3] = a[3]*b00[0];  // 6
      c[4] = a[3]*b01[0]; // 7
      c[5] = a[4]; // 8
      c[6] = a[6]*b00[0]; // 12
      c[7] = a[6]*b01[0]; // 13
      c[8] = a[7]; // 14
      c[9] = a[9]*b00[0]; // 18
      c[10] = a[9]*b01[0]; // 19
      c[11] = a[10]; // 20
      c[12] = a[12]*b00[0]; // 24
      c[13] = a[12]*b01[0]; // 25
      c[14] = a[13]; // 26
      c[15] = a[15]*b00[0]; // 30
      c[16] = a[15]*b01[0]; // 31
      c[17] = a[16]; // 32
}

__device__
void KHC_polar_fn(const GPlexRegLH &a, const GPlexLS &B, GPlexLS &C, const int n)
{
  // C = A * B, C is 6x6, A is 6x6 , B is 6x6 sym
  using T = float;
                 /*int aN = 1; int an = 0;  // Register array*/
  T *b = B.ptr;  int bN = B.stride;  int bn = n;
  T *c = C.ptr;  int cN = C.stride;  int cn = n;
  /*trove::coalesced_ptr<T> b {B.ptr};  int bN = B.stride;  int bn = n;*/
  /*trove::coalesced_ptr<T> c {C.ptr};  int cN = C.stride;  int cn = n;*/
#if 0
#include "KHC.ah"
#else
      T rb[5];

      rb[0] = b[ 0*bN+bn];
      rb[1] = b[ 1*bN+bn];
      rb[2] = b[ 2*bN+bn];
      rb[3] = b[ 3*bN+bn];
      rb[4] = b[ 4*bN+bn];
      c[ 0*cN+cn] = a[ 0]*rb[ 0] + a[ 1]*rb[ 1] + a[ 2]*rb[ 3];
      c[ 1*cN+cn] = a[ 3]*rb[ 0] + a[ 4]*rb[ 1] + a[ 5]*rb[ 3];
      c[ 2*cN+cn] = a[ 3]*rb[ 1] + a[ 4]*rb[ 2] + a[ 5]*rb[ 4];

      c[ 3*cN+cn] = a[6]*rb[ 0] + a[7]*rb[ 1] + a[8]*rb[ 3];
      c[ 4*cN+cn] = a[6]*rb[ 1] + a[7]*rb[ 2] + a[8]*rb[ 4];
      c[ 5*cN+cn] = a[6]*rb[ 3] + a[7]*rb[ 4] + a[8]*b[ 5*bN+bn];

      c[ 6*cN+cn] = a[9]*rb[ 0] + a[10]*rb[ 1] + a[11]*rb[ 3];
      c[ 7*cN+cn] = a[9]*rb[ 1] + a[10]*rb[ 2] + a[11]*rb[ 4];
      c[ 8*cN+cn] = a[9]*rb[ 3] + a[10]*rb[ 4] + a[11]*b[ 5*bN+bn];

      c[10*cN+cn] = a[12]*rb[ 0] + a[13]*rb[ 1] + a[14]*rb[ 3];
      c[11*cN+cn] = a[12]*rb[ 1] + a[13]*rb[ 2] + a[14]*rb[ 4];
      c[15*cN+cn] = a[15]*rb[ 0] + a[16]*rb[ 1] + a[17]*rb[ 3];
      c[16*cN+cn] = a[15]*rb[ 1] + a[16]*rb[ 2] + a[17]*rb[ 4];

      rb[0] = b[ 5*bN+bn];
      c[12*cN+cn] = a[12]*rb[ 3] + a[13]*rb[ 4] + a[14]*rb[ 0];
      c[17*cN+cn] = a[15]*rb[ 3] + a[16]*rb[ 4] + a[17]*rb[ 0];

      rb[0] = b[ 6*bN+bn];
      rb[1] = b[ 7*bN+bn];
      rb[2] = b[ 8*bN+bn];
      c[ 9*cN+cn] = a[9]*rb[ 0] + a[10]*rb[ 1] + a[11]*rb[ 1];
      c[13*cN+cn] = a[12]*rb[ 0] + a[13]*rb[ 1] + a[14]*rb[ 2];
      c[18*cN+cn] = a[15]*rb[ 0] + a[16]*rb[ 1] + a[17]*rb[ 2];

      rb[0] = b[ 10*bN+bn];
      rb[1] = b[ 11*bN+bn];
      rb[2] = b[ 12*bN+bn];
      c[14*cN+cn] = a[12]*rb[ 0] + a[13]*rb[1] + a[14]*rb[2];
      c[19*cN+cn] = a[15]*rb[ 0] + a[16]*rb[1] + a[17]*rb[2];

      c[20*cN+cn] = a[15]*b[15*bN+bn] + a[16]*b[16*bN+bn] + a[17]*b[17*bN+bn];
#endif
}

#ifndef CCSCOORD 
__device__
void ConvertToCCS_fn(const GPlexLV &a, GPlexRegLV &b, GPlexRegLL &c, const int n)
{
  ConvertToCCS_imp(a, b, c, n, n+1);
}
#endif

__device__
void PolarErr_fn(const GPlexRegLL &a, const float *b, int bN, GPlexRegLL &c, const int n)
{
  // C = A * B, C is 6x6, A is 6x6 , B is 6x6 sym
 
  // Generated code access arrays with variables cN, cn
  // c[i*cN+cn]  
  int aN = 1; int an = 0;  // Register array
              int bn = n;  // Global array
  int cN = 1; int cn = 0;
#if 0
#include "CCSErr.ah"
#else
  c[ 0*cN+cn] = b[ 0*bN+bn];

  c[ 1*cN+cn] = b[ 1*bN+bn];
  c[ 6*cN+cn] = b[ 1*bN+bn];

  c[ 7*cN+cn] = b[ 2*bN+bn];

  c[ 2*cN+cn] = b[ 3*bN+bn];
  c[12*cN+cn] = b[ 3*bN+bn];

  c[ 8*cN+cn] = b[ 4*bN+bn];
  c[13*cN+cn] = b[ 4*bN+bn];

  c[14*cN+cn] = b[ 5*bN+bn];

  c[ 3*cN+cn] = b[ 6*bN+bn];
  c[18*cN+cn] = a[21*aN+an]*b[ 6*bN+bn];
  c[24*cN+cn] = a[27*aN+an]*b[ 6*bN+bn];
  c[30*cN+cn] = a[33*aN+an]*b[ 6*bN+bn];

  c[ 9*cN+cn] = b[ 7*bN+bn];
  c[19*cN+cn] = a[21*aN+an]*b[ 7*bN+bn];
  c[25*cN+cn] = a[27*aN+an]*b[ 7*bN+bn];
  c[31*cN+cn] = a[33*aN+an]*b[ 7*bN+bn];

  c[15*cN+cn] = b[ 8*bN+bn];
  c[20*cN+cn] = a[21*aN+an]*b[ 8*bN+bn];
  c[26*cN+cn] = a[27*aN+an]*b[ 8*bN+bn];
  c[32*cN+cn] = a[33*aN+an]*b[ 8*bN+bn];

  c[21*cN+cn] = a[21*aN+an]*b[ 9*bN+bn];
  c[27*cN+cn] = a[27*aN+an]*b[ 9*bN+bn];
  c[33*cN+cn] = a[33*aN+an]*b[ 9*bN+bn];

  c[ 4*cN+cn] = b[10*bN+bn];
  c[18*cN+cn] += a[22*aN+an]*b[10*bN+bn];
  c[24*cN+cn] += a[28*aN+an]*b[10*bN+bn];
  c[30*cN+cn] += a[34*aN+an]*b[10*bN+bn];

  c[10*cN+cn] = b[11*bN+bn];
  c[19*cN+cn] += a[22*aN+an]*b[11*bN+bn];
  c[25*cN+cn] += a[28*aN+an]*b[11*bN+bn];
  c[31*cN+cn] += a[34*aN+an]*b[11*bN+bn];

  c[16*cN+cn] = b[12*bN+bn];
  c[20*cN+cn] += a[22*aN+an]*b[12*bN+bn];
  c[26*cN+cn] += a[28*aN+an]*b[12*bN+bn];
  c[32*cN+cn] += a[34*aN+an]*b[12*bN+bn];

  c[21*cN+cn] += a[22*aN+an]*b[13*bN+bn];
  c[22*cN+cn] = a[21*aN+an]*b[13*bN+bn];
  c[27*cN+cn] += a[28*aN+an]*b[13*bN+bn];
  c[28*cN+cn] = a[27*aN+an]*b[13*bN+bn];
  c[33*cN+cn] += a[34*aN+an]*b[13*bN+bn];
  c[34*cN+cn] = a[33*aN+an]*b[13*bN+bn];

  c[22*cN+cn] += a[22*aN+an]*b[14*bN+bn];
  c[28*cN+cn] += a[28*aN+an]*b[14*bN+bn];
  c[34*cN+cn] += a[34*aN+an]*b[14*bN+bn];

  c[ 5*cN+cn] = b[15*bN+bn];
  c[30*cN+cn] += a[35*aN+an]*b[15*bN+bn];

  c[11*cN+cn] = b[16*bN+bn];
  c[31*cN+cn] += a[35*aN+an]*b[16*bN+bn];

  c[17*cN+cn] = b[17*bN+bn];
  c[32*cN+cn] += a[35*aN+an]*b[17*bN+bn];

  c[23*cN+cn] = a[21*aN+an]*b[18*bN+bn];
  c[29*cN+cn] = a[27*aN+an]*b[18*bN+bn];
  c[33*cN+cn] += a[35*aN+an]*b[18*bN+bn];
  c[35*cN+cn] = a[33*aN+an]*b[18*bN+bn];

  c[23*cN+cn] += a[22*aN+an]*b[19*bN+bn];
  c[29*cN+cn] += a[28*aN+an]*b[19*bN+bn];
  c[34*cN+cn] += a[35*aN+an]*b[19*bN+bn];
  c[35*cN+cn] += a[34*aN+an]*b[19*bN+bn];

  c[35*cN+cn] += a[35*aN+an]*b[20*bN+bn];
#endif
}

#ifndef CCSCOORD
__device__
void PolarErrTransp_fn(const GPlexRegLL &a, const GPlexRegLL &b, GPlexLS &C, const int n)
{
  // C = A * B, C is sym, A is 6x6 , B is 6x6
  using T = float;
                 int aN = 1;         int an = 0;
                 int bN = 1;         int bn = 0;
  T *c = C.ptr;  int cN = C.stride;  int cn = n;
#include "CCSErrTransp.ah"
}
#endif

#ifndef CCSCOORD
__device__
void ConvertToCartesian_fn(const GPlexRegLV &a, GPlexLV& b, GPlexRegLL &c, const int n)
{
  ConvertToCartesian_imp(a, b, c, n, n+1); 
}
#endif

__device__
void CartesianErr_fn(const GPlexRegLL &a, const float *b, const int bN, GPlexRegLL &c, const int n)
{
  // C = A * B, C is 6x6, A is 6x6 , B is 6x6 sym
  int aN = 1; int an = 0;
              int bn = n;
  int cN = 1; int cn = 0;

#if 0
#include "CartesianErr.ah"
#else
  c[ 0*cN+cn] = b[ 0*bN+bn];

  c[ 6*cN+cn] = b[ 1*bN+bn];
  c[ 1*cN+cn] = b[ 1*bN+bn];

  c[ 7*cN+cn] = b[ 2*bN+bn];
  
  c[ 2*cN+cn] = b[ 3*bN+bn];
  c[12*cN+cn] = b[ 3*bN+bn];

  c[ 8*cN+cn] = b[ 4*bN+bn];
  c[13*cN+cn] = b[ 4*bN+bn];

  c[14*cN+cn] = b[ 5*bN+bn];

  c[ 3*cN+cn] = b[ 6*bN+bn];
  c[18*cN+cn] = a[21*aN+an]*b[ 6*bN+bn];
  c[24*cN+cn] = a[27*aN+an]*b[ 6*bN+bn];
  c[30*cN+cn] = a[33*aN+an]*b[ 6*bN+bn];

  c[ 9*cN+cn] = b[ 7*bN+bn];
  c[19*cN+cn] = a[21*aN+an]*b[ 7*bN+bn];
  c[25*cN+cn] = a[27*aN+an]*b[ 7*bN+bn];
  c[31*cN+cn] = a[33*aN+an]*b[ 7*bN+bn];
  
  c[15*cN+cn] = b[ 8*bN+bn];
  c[20*cN+cn] = a[21*aN+an]*b[ 8*bN+bn];
  c[26*cN+cn] = a[27*aN+an]*b[ 8*bN+bn];
  c[32*cN+cn] = a[33*aN+an]*b[ 8*bN+bn];

  c[21*cN+cn] = a[21*aN+an]*b[ 9*bN+bn];
  c[27*cN+cn] = a[27*aN+an]*b[ 9*bN+bn];
  c[33*cN+cn] = a[33*aN+an]*b[ 9*bN+bn];

  c[ 4*cN+cn] = b[10*bN+bn];
  c[18*cN+cn] += a[22*aN+an]*b[10*bN+bn];
  c[24*cN+cn] += a[28*aN+an]*b[10*bN+bn];

  c[10*cN+cn] = b[11*bN+bn];
  c[19*cN+cn] += a[22*aN+an]*b[11*bN+bn];
  c[25*cN+cn] += a[28*aN+an]*b[11*bN+bn];

  c[16*cN+cn] = b[12*bN+bn];
  c[20*cN+cn] += a[22*aN+an]*b[12*bN+bn];
  c[26*cN+cn] += a[28*aN+an]*b[12*bN+bn];

  c[21*cN+cn] += a[22*aN+an]*b[13*bN+bn];
  c[22*cN+cn] = a[21*aN+an]*b[13*bN+bn];
  c[27*cN+cn] += a[28*aN+an]*b[13*bN+bn];
  c[28*cN+cn] = a[27*aN+an]*b[13*bN+bn];
  c[34*cN+cn] = a[33*aN+an]*b[13*bN+bn];

  c[22*cN+cn] += a[22*aN+an]*b[14*bN+bn];
  c[28*cN+cn] += a[28*aN+an]*b[14*bN+bn];

  c[ 5*cN+cn] = b[15*bN+bn];
  c[30*cN+cn] += a[35*aN+an]*b[15*bN+bn];

  c[11*cN+cn] = b[16*bN+bn];
  c[31*cN+cn] += a[35*aN+an]*b[16*bN+bn];

  c[17*cN+cn] = b[17*bN+bn];
  c[32*cN+cn] += a[35*aN+an]*b[17*bN+bn];

  c[23*cN+cn] = a[21*aN+an]*b[18*bN+bn];
  c[29*cN+cn] = a[27*aN+an]*b[18*bN+bn];
  c[33*cN+cn] += a[35*aN+an]*b[18*bN+bn];
  c[35*cN+cn] = a[33*aN+an]*b[18*bN+bn];

  c[23*cN+cn] += a[22*aN+an]*b[19*bN+bn];
  c[29*cN+cn] += a[28*aN+an]*b[19*bN+bn];
  c[34*cN+cn] += a[35*aN+an]*b[19*bN+bn];

  c[35*cN+cn] += a[35*aN+an]*b[20*bN+bn];
#endif
}

__device__
void CartesianErrTransp_fn(const GPlexRegLL &a, const GPlexRegLL &b, GPlexLS &C, const int n)
{
  // C = A * B, C is sym, A is 6x6 , B is 6x6
  using T = float;
  int aN = 1; int an = 0;
  int bN = 1; int bn = 0;
  T *c = C.ptr;  int cN = C.stride;  int cn = n;

#include "CartesianErrTransp.ah"
}


/// MultKalmanGain ////////////////////////////////////////////////////////////

__device__ void upParam_MultKalmanGain_fn(
    const float* __restrict__ a, const size_t aN,
    const float* b_reg, float *c, const int N, const int n) {
  // const T* __restrict__ tells the compiler that it can uses the read-only
  // cache, without worrying about coherency.
  // c -> kalmanGain, in register

  /*int n = threadIdx.x + blockIdx.x * blockDim.x;*/
  // use registers to store values of 'a' that are use multiple times
  // To reduce the number of registers and avoid spilling interlace
  // read and compute instructions.
  float a_reg[3];

  int j;
  j = 0; a_reg[0] = a[n + j*aN];
  j = 1; a_reg[1] = a[n + j*aN];
  j = 3; a_reg[2] = a[n + j*aN];

  c[ 0] = a_reg[ 0]*b_reg[ 0] + a_reg[ 1]*b_reg[ 1] + a_reg[ 2]*b_reg[ 3];
  c[ 1] = a_reg[ 0]*b_reg[ 1] + a_reg[ 1]*b_reg[ 2] + a_reg[ 2]*b_reg[ 4];
  c[ 2] = a_reg[ 0]*b_reg[ 3] + a_reg[ 1]*b_reg[ 4] + a_reg[ 2]*b_reg[ 5];

  j = 1; a_reg[0] = a[n + j*aN];
  j = 2; a_reg[1] = a[n + j*aN];
  j = 4; a_reg[2] = a[n + j*aN];

  c[ 3] = a_reg[ 0]*b_reg[ 0] + a_reg[ 1]*b_reg[ 1] + a_reg[ 2]*b_reg[ 3];
  c[ 4] = a_reg[ 0]*b_reg[ 1] + a_reg[ 1]*b_reg[ 2] + a_reg[ 2]*b_reg[ 4];
  c[ 5] = a_reg[ 0]*b_reg[ 3] + a_reg[ 1]*b_reg[ 4] + a_reg[ 2]*b_reg[ 5];

  j = 3; a_reg[0] = a[n + j*aN];
  j = 4; a_reg[1] = a[n + j*aN];
  j = 5; a_reg[2] = a[n + j*aN];

  c[ 6] = a_reg[ 0]*b_reg[ 0] + a_reg[ 1]*b_reg[ 1] + a_reg[ 2]*b_reg[ 3];
  c[ 7] = a_reg[ 0]*b_reg[ 1] + a_reg[ 1]*b_reg[ 2] + a_reg[ 2]*b_reg[ 4];
  c[ 8] = a_reg[ 0]*b_reg[ 3] + a_reg[ 1]*b_reg[ 4] + a_reg[ 2]*b_reg[ 5];

  j = 6; a_reg[0] = a[n + j*aN];
  j = 7; a_reg[1] = a[n + j*aN];
  j = 8; a_reg[2] = a[n + j*aN];

  c[ 9] = a_reg[ 0]*b_reg[ 0] + a_reg[ 1]*b_reg[ 1] + a_reg[ 2]*b_reg[ 3];
  c[10] = a_reg[ 0]*b_reg[ 1] + a_reg[ 1]*b_reg[ 2] + a_reg[ 2]*b_reg[ 4];
  c[11] = a_reg[ 0]*b_reg[ 3] + a_reg[ 1]*b_reg[ 4] + a_reg[ 2]*b_reg[ 5];

  j = 10; a_reg[0] = a[n + j*aN];
  j = 11; a_reg[1] = a[n + j*aN];
  j = 12; a_reg[2] = a[n + j*aN];

  c[12] = a_reg[ 0]*b_reg[ 0] + a_reg[ 1]*b_reg[ 1] + a_reg[ 2]*b_reg[ 3];
  c[13] = a_reg[ 0]*b_reg[ 1] + a_reg[ 1]*b_reg[ 2] + a_reg[ 2]*b_reg[ 4];
  c[14] = a_reg[ 0]*b_reg[ 3] + a_reg[ 1]*b_reg[ 4] + a_reg[ 2]*b_reg[ 5];

  j = 15; a_reg[0] = a[n + j*aN];
  j = 16; a_reg[1] = a[n + j*aN];
  j = 17; a_reg[2] = a[n + j*aN];

  c[15] = a_reg[ 0]*b_reg[ 0] + a_reg[ 1]*b_reg[ 1] + a_reg[ 2]*b_reg[ 3];
  c[16] = a_reg[ 0]*b_reg[ 1] + a_reg[ 1]*b_reg[ 2] + a_reg[ 2]*b_reg[ 4];
  c[17] = a_reg[ 0]*b_reg[ 3] + a_reg[ 1]*b_reg[ 4] + a_reg[ 2]*b_reg[ 5];
}

/// Invert Cramer Symetric ////////////////////////////////////////////////////
#ifndef CCSCOORD
__device__ void invertCramerSym_fn(float *a) {
  // a is in registers.
  // to use global memory, a stride will be required and accesses would be:
  // a[n + stride_a * i];
  typedef float TT;

  const TT c00 = a[2] * a[5] - a[4] * a[4];
  const TT c01 = a[4] * a[3] - a[1] * a[5];
  const TT c02 = a[1] * a[4] - a[2] * a[3];
  const TT c11 = a[5] * a[0] - a[3] * a[3];
  const TT c12 = a[3] * a[1] - a[4] * a[0];
  const TT c22 = a[0] * a[2] - a[1] * a[1];

  const TT det = a[0] * c00 + a[1] * c01 + a[3] * c02;

  const TT s = TT(1) / det;

  a[0] = s*c00;
  a[1] = s*c01;
  a[2] = s*c11;
  a[3] = s*c02;
  a[4] = s*c12;
  a[5] = s*c22;
}
#endif

__device__ void invertCramerSym2x2_fn(GPlexReg2S &a) {
#if 0
  float det = a[0] * a[2] - a[1] * a[1];
  const float s   = 1.f / det;
  const float tmp = s * a[2];
  a[1] *= -s;
  a[2]  = s * a[0];
  a[0]  = tmp;
#else
  float det = __fmaf_rn( a[0] , a[2] , __fmul_rn(a[1], a[1]) );
  /*const float s   = 1.f / det;*/
  const float tmp = __fdividef( a[2] , det);
  a[1] *= - __fdividef(1.f, det);
  a[2]  = __fdividef(a[0], det);
  a[0]  = tmp;
#endif
}

__device__ void subtractFirst3_fn(const GPlexHV __restrict__ &A,
                                  const GPlexLV __restrict__ &B,
                                  GPlexRegHV &C, const int N, int n) {
  using T = float;
  const T *a = A.ptr;  int aN = A.stride;
  const T *b = B.ptr;  int bN = B.stride;
        T *c = C.arr;
  /*int n = threadIdx.x + blockIdx.x * blockDim.x;*/
  
  if(n < N) {
    c[0] = a[0*aN+n] - b[0*bN+n];
    c[1] = a[1*aN+n] - b[1*bN+n];
    c[2] = a[2*aN+n] - b[2*bN+n];
  }
}

__device__ void subtract3x3_rotate_fn(const GPlexHV __restrict__ &A,
                                      const GPlexLV __restrict__ &B,
                                      GPlexRegQF& r00, GPlexRegQF& r01,
                                      GPlexReg2V& res,
                                      int n, int N)
{
  using T = float;
  const T *a = A.ptr;  int aN = A.stride;
  const T *b = B.ptr;  int bN = B.stride;
  /*int n = threadIdx.x + blockIdx.x * blockDim.x;*/
  float c[3];
  
  if(n < N) {
    c[0] = a[0*aN+n] - b[0*bN+n];
    c[1] = a[1*aN+n] - b[1*bN+n];
    c[2] = a[2*aN+n] - b[2*bN+n];

    res(0, 0, 0) =  r00(0, 0, 0)*c[0] + r01(0, 0, 0)*c[1];
    res(0, 0, 1) =  c[2];
   }
}

/// AddIntoUpperLeft3x3  //////////////////////////////////////////////////////
__device__ void addIntoUpperLeft3x3_fn(const GPlexLS __restrict__ &A,
                                       const GPlexHS __restrict__ &B,
                                       GPlexRegHS &C, const int N, const int n) {
  using T = float;
  const T *a = A.ptr;  int aN = A.stride;
  const T *b = B.ptr;  int bN = B.stride;
        T *c = C.arr;
  /*int n = threadIdx.x + blockIdx.x * blockDim.x;*/
  
  if(n < N) {
    c[0] = a[0*aN+n] + b[0*bN+n];
    c[1] = a[1*aN+n] + b[1*bN+n];
    c[2] = a[2*aN+n] + b[2*bN+n];
    c[3] = a[3*aN+n] + b[3*bN+n];
    c[4] = a[4*aN+n] + b[4*bN+n];
    c[5] = a[5*aN+n] + b[5*bN+n];
  }
}


/// MultResidualsAdd //////////////////////////////////////////////////////////
__device__ void multResidualsAdd_fn(
    const GPlexRegLH &reg_a,
    const GPlexLV __restrict__ &B, 
    const GPlexReg2V &c,
          GPlexLV &D,
    const int N, const int n) {

  MultResidualsAdd_imp(reg_a, B, c, D, n, n+1);
}

__device__
void MultResidualsAdd_all_reg(const GPlexRegLH &a,
		      const GPlexRegLV &b,
		      const GPlexReg2V &c,
          GPlexRegLV &d)
{
   // outPar = psPar + kalmanGain*(dPar)
   //   D    =   B         A         C
   // where right half of kalman gain is 0 

   // XXX Regenerate with a script.
      // generate loop (can also write it manually this time, it's not much)
      d[0] = b[0] + a[ 0] * c[0] + a[ 1] * c[1];
      d[1] = b[1] + a[ 3] * c[0] + a[ 4] * c[1];
      d[2] = b[2] + a[ 6] * c[0] + a[ 7] * c[1];
      d[3] = b[3] + a[ 9] * c[0] + a[10] * c[1];
      d[4] = b[4] + a[12] * c[0] + a[13] * c[1];
      d[5] = b[5] + a[15] * c[0] + a[16] * c[1];
}


__device__ void kalmanUpdate_fn(
    GPlexLS &propErr, const GPlexHS __restrict__ &msErr,
    const GPlexLV __restrict__ &par_iP, const GPlexHV __restrict__ &msPar,
    GPlexLV &par_iC, GPlexLS &outErr, const int n, const int N) {
  // If there is more matrices than max_blocks_x * BLOCK_SIZE_X 
  if (n < N) {
    // FIXME: Add useCMSGeom -> port propagateHelixToRMPlex
#if 0
    if (Config::useCMSGeom) {
      propagateHelixToRMPlex(psErr,  psPar, inChg,  msPar, propErr, propPar);
    } else {
      propErr = psErr;
      propPar = psPar;
    }
#endif
    GPlexRegQF rotT00;
    GPlexRegQF rotT01;
    const float r = hipo(msPar(n, 0, 0), msPar(n, 1, 0));
    rotT00[0] = -(msPar(n, 1, 0) + par_iP(n, 1, 0))/(2*r);
    rotT01[0] =  (msPar(n, 0, 0) + par_iP(n, 0, 0))/(2*r);

    GPlexRegHS resErr_reg;
    addIntoUpperLeft3x3_fn(propErr, msErr, resErr_reg, N, n);

    GPlexRegHH tempHH;  // 3*3 sym
    ProjectResErr_fn  (rotT00, rotT01, resErr_reg, tempHH);

    GPlexReg2S resErr_loc; // 2x2 sym
    ProjectResErrTransp_fn(rotT00, rotT01, tempHH, resErr_loc);

    invertCramerSym2x2_fn(resErr_loc);
#ifndef CCSCOORD
    // Move to "polar" coordinates: (x,y,z,1/pT,phi,theta) [can we find a better name?]

    GPlexRegLV propPar_pol;  // propagated parameters in "polar" coordinates*/
    GPlexRegLL jac_pol;  // jacobian from cartesian to "polar"*/

    ConvertToCCS_fn(par_iP, propPar_pol, jac_pol, n);

    GPlexRegLL tempLL;
    PolarErr_fn(jac_pol, propErr.ptr, propErr.stride, tempLL, n);
    PolarErrTransp_fn(jac_pol, tempLL, propErr, n);// propErr is now propagated errors in "polar" coordinates
#endif

    // Kalman update in "polar" coordinates
    GPlexRegLH K;
    KalmanHTG_fn(rotT00, rotT01, resErr_loc, tempHH);
    KalmanGain_fn(propErr, tempHH, K, n);

    GPlexReg2V res_loc;   //position residual in local coordinates
    subtract3x3_rotate_fn(msPar, par_iP, rotT00, rotT01, res_loc, n, N);

#ifdef CCSCOORD
    multResidualsAdd_fn(K, par_iP, res_loc, par_iC, N, n);// propPar_pol is now the updated parameters in "polar" coordinates
    /*GPlexRegLH tempLL;  // LL as a LH: half of the values are 0.*/
    GPlexRegLL tempLL;  // LL as a LH: half of the values are 0.

    KHMult_fn(K, rotT00, rotT01, tempLL);
    KHC_fn(tempLL, propErr, outErr, n);
#else
    MultResidualsAdd_all_reg(K, propPar_pol, res_loc, propPar_pol);
    KHMult_fn(K, rotT00, rotT01, tempLL);
    KHC_fn(tempLL, propErr, outErr, n);
#endif

    subtract_matrix(propErr.ptr, propErr.stride, outErr.ptr, outErr.stride, 
                    outErr.ptr, outErr.stride, LS, n);

#ifndef CCSCOORD
    // Go back to cartesian coordinates

    // jac_pol is now the jacobian from "polar" to cartesian
    // outPar -> par_iC
    ConvertToCartesian_fn(propPar_pol, par_iC, jac_pol, n);
    CartesianErr_fn      (jac_pol, outErr.ptr, outErr.stride, tempLL, n);
    CartesianErrTransp_fn(jac_pol, tempLL, outErr, n);// outErr is in cartesian coordinates now
#endif
  }
}

__global__ void kalmanUpdate_kernel(
    GPlexLS propErr, const GPlexHS __restrict__ msErr,
    const GPlexLV __restrict__ par_iP, const GPlexHV __restrict__ msPar,
    GPlexLV par_iC, GPlexLS outErr, const int N) {
  int grid_width = blockDim.x * gridDim.x;
  int n = threadIdx.x + blockIdx.x * blockDim.x;

  for (int z = 0; z < (N-1)/grid_width  +1; z++) {
    n += z*grid_width;
    kalmanUpdate_fn(propErr, msErr, par_iP, msPar, par_iC, outErr, n, N);
  }
}

void kalmanUpdate_wrapper(const cudaStream_t& stream,
    GPlexLS& d_propErr, const GPlexHS& d_msErr,
    GPlexLV& d_par_iP, const GPlexHV& d_msPar,
    GPlexLV& d_par_iC, GPlexLS& d_outErr,
    const int N) {
  int gridx = std::min((N-1)/BLOCK_SIZE_X + 1,
                       max_blocks_x);
  dim3 grid(gridx, 1, 1);
  dim3 block(BLOCK_SIZE_X, 1, 1);
  /*kalmanUpdate_kernel <<<grid, block, 0, stream >>>*/
      /*(d_propErr, d_msErr, d_par_iP, d_msPar, d_par_iC, d_outErr, N);*/
}

