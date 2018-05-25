#include "computeChi2_kernels.h"

#include "GPlex.h"
#include "kalmanUpdater_kernels.h"
#include "gpu_utils.h"

#include <cstdio>

#define L 6
#define HS 6
#define HV 3
#define BLOCK_SIZE_X 256

#include "KalmanUtilsMPlex.icc"

__device__ float chi2Similarity_fn(
    const GPlexReg2V &a,
    const GPlexReg2S &c)
{
#if 1
  return c[0]*a[0]*a[0]
       + c[2]*a[1]*a[1] 
       + 2*( c[1]*a[1]*a[0]);
#else
  float res;
  asm("{\n\t"  // braces for local scopr
      ".reg .f32 t1;\n\t"   // temp reg t1
      "mul.f32 t1, %1, %1;\n\t" // t1 = a0*a0
      "mul.f32 t1, t1, %3;\n\t" // t1 = a0*c0
      ".reg .f32 t2;\n\t"   // temp reg t2
      "mul.f32 t2, %2, %2;\n\t" // t2 = a1*a1
      "fma.rn.f32 t2, %5, t2, t1;\n\t" // t2 = t2 * c2 + t1
      ".reg .f32 t3;\n\t"   // temp reg t3
      "mul.f32 t3, %2, %1;\n\t" // t3 = a1*a0
      "mul.f32 t3, t3, %4;\n\t" // t3 = t3*c1  
      "fma.rn.f32 %0, t3, %6, t2;"
      "}"
      : "=f"(res)
      : "f"(a[0]), "f"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(2.f)
     );
  return res;
#endif
}


__device__ void RotateResidulsOnTangentPlane_fn(const GPlexRegQF& r00,//r00
				  const GPlexRegQF& r01,//r01
				  const GPlexRegHV &a  ,//res_glo
          GPlexReg2V &b  )//res_loc
{
   // res_loc = rotT * res_glo
   //   B     = N R   *    A   
   RotateResidulsOnTangentPlane_impl(r00, r01, a, b, 0, 1);
}


__device__ void ProjectResErr_fn(const GPlexRegQF& a00,
		   const GPlexRegQF& a01,
		   const GPlexRegHS &b, 
       GPlexRegHH &c)
{
  // C = A * B, C is 3x3, A is 3x3 , B is 3x3 sym

  // Based on script generation and adapted to custom sizes.
      c[ 0] = a00[0]*b[ 0] + a01[0]*b[ 1];
      c[ 1] = a00[0]*b[ 1] + a01[0]*b[ 2];
      c[ 2] = a00[0]*b[ 3] + a01[0]*b[ 4];
      c[ 3] = b[ 3];
      c[ 4] = b[ 4];
      c[ 5] = b[ 5];
      c[ 6] = a01[0]*b[ 0] - a00[0]*b[ 1];
      c[ 7] = a01[0]*b[ 1] - a00[0]*b[ 2];
      c[ 8] = a01[0]*b[ 3] - a00[0]*b[ 4];
}


__device__ void ProjectResErrTransp_fn(const GPlexRegQF& a00,
			 const GPlexRegQF& a01, const GPlexRegHH& b, GPlexReg2S& c)
{
  // C = A * B, C is 3x3 sym, A is 3x3 , B is 3x3

  // Based on script generation and adapted to custom sizes.
      c[ 0] = b[ 0]*a00[0] + b[ 1]*a01[0];
      c[ 1] = b[ 3]*a00[0] + b[ 4]*a01[0];
      c[ 2] = b[ 5];
}


__device__ float computeChi2_fn(
    const GPlexLS &propErr, const GPlexHS &msErr, const GPlexHV &msPar,
    const GPlexLV &propPar, const int n, const int N) {
  //int n = threadIdx.x + blockIdx.x * blockDim.x;

  if (n < N) {
    // coordinate change
    GPlexRegQF rotT00;
    GPlexRegQF rotT01;
    float x = msPar(n, 0, 0);
    float y = msPar(n, 1, 0);
    const float r = hipo(x, y);
    rotT00[0] = -(y + propPar(n, 1, 0))/(2*r);
    rotT01[0] =  (x + propPar(n, 0, 0))/(2*r);

    GPlexRegHV res_glo;
    subtractFirst3_fn(msPar, propPar, res_glo, N, n);

    GPlexReg2V res_loc;   //position residual in local coordinates
    RotateResidulsOnTangentPlane_fn(rotT00,rotT01,res_glo,res_loc);

    GPlexRegHS resErr_reg;
    addIntoUpperLeft3x3_fn(propErr, msErr, resErr_reg, N, n);
    GPlexReg2S resErr_loc; // 2x2 sym
    GPlexRegHH tempHH;  // 3*3 sym
    ProjectResErr_fn  (rotT00, rotT01, resErr_reg, tempHH);
    ProjectResErrTransp_fn(rotT00, rotT01, tempHH, resErr_loc);

    invertCramerSym2x2_fn(resErr_loc);

    return chi2Similarity_fn(res_loc, resErr_loc);
  }
  return 0;
}


__global__ void computeChi2_kernel(
    const GPlexLS propErr, const GPlexHS msErr, const GPlexHV msPar, 
    const GPlexLV propPar, const int N) {
  int grid_width = blockDim.x * gridDim.x;
  int itrack = threadIdx.x + blockDim.x*blockIdx.x;
  for (int z = 0; z < (N-1)/grid_width  +1; z++) {
    itrack += z*grid_width;

    if (itrack < N) {
      computeChi2_fn (propErr, msErr, msPar, propPar, itrack, N);
    }
  }
}


void computeChi2_wrapper(cudaStream_t &stream, 
    const GPlexLS &propErr, const GPlexHS &msErr,
    const GPlexHV &msPar, const GPlexLV &propPar, 
    const int N) {
  int gridx = std::min((N-1)/BLOCK_SIZE_X + 1,
                       max_blocks_x);
  dim3 grid(gridx, 1, 1);
  dim3 block(BLOCK_SIZE_X, 1, 1);
  computeChi2_kernel <<< grid, block, 0, stream >>>
    (propErr, msErr, msPar, propPar, N);
 }
