#include "build_tracks_kernels.h"

#include "reorganize_gplex.h"
#include "kalmanUpdater_kernels.h"
#include "computeChi2_kernels.h"

constexpr int BLOCK_SIZE_X = 16;


__device__ void getHitFromLayer_fn(LayerOfHitsCU& layer_of_hits, 
    GPlexQI& HitsIdx, GPlexHV& msPar, GPlexHS& msErr, int itrack_plex, int N)
{
  if (itrack_plex < N)
  {
    int hit_idx = HitsIdx[itrack_plex];
    if (hit_idx >= 0)
    {
      Hit &hit = layer_of_hits.m_hits[hit_idx];

      GetHitErr(msErr, (char *)hit.errArrayCU(), 0, N);
      GetHitPar(msPar, (char *)hit.posArrayCU(), 0, N);
    }
  }
}

__global__ void getHitFromLayer_kernel(LayerOfHitsCU& layer_of_hits, 
    GPlexQI HitsIdx, GPlexHV msPar, GPlexHS msErr, int N)
{
  int itrack_plex = threadIdx.x + blockDim.x * blockIdx.x;
  getHitFromLayer_fn(layer_of_hits, HitsIdx, msPar, msErr, itrack_plex, N);
}

void getHitFromLayer_wrappper( const cudaStream_t& stream,
    LayerOfHitsCU& layer_cu, GPlexQI& HitsIdx, 
    GPlexHV& msPar, GPlexHS& msErr, int N)
{
  int gridx = (N-1)/BLOCK_SIZE_X + 1;
  dim3 grid(gridx, 1, 1);
  dim3 block(BLOCK_SIZE_X, 1, 1);

  getHitFromLayer_kernel <<< grid, block, 0, stream >>>
    (layer_cu, HitsIdx, msPar, msErr, N);
}




__device__ void updateMissingHits_fn(GPlexQI& HitsIdx, 
    GPlexLV& Par_iP, GPlexLS& Err_iP,
    GPlexLV& Par_iC, GPlexLS& Err_iC, int i, int N)
{
  if (i < N)
  {
    if (HitsIdx[i] < 0)
    {
      two_steps_copy<7> (Err_iC, Err_iP, i);
      two_steps_copy<6> (Par_iC, Par_iP, i);
    }
  }
}


__global__ void updateMissingHits_kernel(GPlexQI HitsIdx, 
    GPlexLV Par_iP, GPlexLS Err_iP,
    GPlexLV Par_iC, GPlexLS Err_iC, int N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  updateMissingHits_fn(HitsIdx, Par_iP, Err_iP, Par_iC, Err_iC, i, N);
}


void UpdateMissingHits_wrapper(
    const cudaStream_t& stream, GPlexQI& HitsIdx, 
    GPlexLV& Par_iP, GPlexLS& Err_iP,
    GPlexLV& Par_iC, GPlexLS& Err_iC,
    int N)
{
  int gridx = (N-1)/BLOCK_SIZE_X + 1;
  dim3 grid(gridx, 1, 1);
  dim3 block(BLOCK_SIZE_X, 1, 1);

  updateMissingHits_kernel <<< grid, block, 0, stream >>>
    (HitsIdx, Par_iP, Err_iP, Par_iC, Err_iC, N);
}


__device__
void UpdateWithLastHit_fn(
    LayerOfHitsCU& layer_of_hits, GPlexQI& HitsIdx, 
    GPlexHV& msPar,  GPlexHS& msErr,
    GPlexLV& Par_iP, GPlexLS& Err_iP,
    GPlexLV& Par_iC, GPlexLS& Err_iC,
    int itrack_plex, int N)
{
  getHitFromLayer_fn(layer_of_hits, HitsIdx, msPar, msErr, itrack_plex, N);
  kalmanUpdate_fn(Err_iP, msErr, Par_iP, msPar, Par_iC, Err_iC, itrack_plex, N);
  updateMissingHits_fn(HitsIdx, Par_iP, Err_iP, Par_iC, Err_iC, itrack_plex, N);
}


__global__
void UpdateWithLastHit_kernel(
    LayerOfHitsCU& layer_of_hits, GPlexQI HitsIdx, 
    GPlexHV msPar, GPlexHS msErr,
    GPlexLV Par_iP, GPlexLS Err_iP,
    GPlexLV Par_iC, GPlexLS Err_iC,
    int N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  UpdateWithLastHit_fn(layer_of_hits, HitsIdx, msPar, msErr, 
                       Par_iP, Err_iP, Par_iC, Err_iC, i, N);
}


void UpdateWithLastHit_wrapper(
    const cudaStream_t& stream,
    LayerOfHitsCU& layer_cu, GPlexQI& HitsIdx, 
    GPlexHV& msPar, GPlexHS& msErr,
    GPlexLV& Par_iP, GPlexLS& Err_iP,
    GPlexLV& Par_iC, GPlexLS& Err_iC,
    int N)
{
  int gridx = (N-1)/BLOCK_SIZE_X + 1;
  dim3 grid(gridx, 1, 1);
  dim3 block(BLOCK_SIZE_X, 1, 1);

  UpdateWithLastHit_kernel <<< grid, block, 0, stream >>>
    (layer_cu, HitsIdx, msPar, msErr,
     Par_iP, Err_iP, Par_iC, Err_iC, N);
}
