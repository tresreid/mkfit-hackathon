#include <cstdlib>
#include "Config.h"
#include "GeometryCU.h"
#include "reorganize_gplex.h"
#include "fittracks_kernels.h"
//#include "build_tracks_kernels.h"

//#include "clone_engine_kernels.h"
//#include "clone_engine_kernels_seed.h"

#include "Track.h"


template <typename T>
void FitterCU<T>::setNumberTracks(const idx_t Ntracks) {
  N = Ntracks;

  // Raise an exceptioin when the FitterCU instance is too small
  // This should not happen as the loop over tracks in runFittestGPU 
  // takes care of it.
  if (Ntracks > Nalloc) {
    throw std::length_error("FitterCU: Ntracks should be smaller than Nalloc");
  }
}

template <typename T>
void FitterCU<T>::createStream() {
  cudaStreamCreate(&stream);
}

template <typename T>
void FitterCU<T>::destroyStream() {
  cudaStreamDestroy(stream);
}

template <typename T>
void FitterCU<T>::allocateDevice() {
  d_par_iP.allocate(Nalloc);
  d_par_iC.allocate(Nalloc);

  d_Err_iP.allocate(Nalloc);
  d_Err_iC.allocate(Nalloc);

  d_inChg.allocate(Nalloc);
  d_errorProp.allocate(Nalloc);

  cudaMalloc((void**)&d_msPar_arr, Config::nLayers * sizeof(GPlexHV));
  cudaMalloc((void**)&d_msErr_arr, Config::nLayers * sizeof(GPlexHS));
  for (int hi = 0; hi < Config::nLayers; ++hi) {
    d_msPar[hi].allocate(Nalloc);
    d_msErr[hi].allocate(Nalloc);
  }
  cudaMemcpy(d_msPar_arr, d_msPar, Config::nLayers*sizeof(GPlexHV), cudaMemcpyHostToDevice);
  cudaMemcpy(d_msErr_arr, d_msErr, Config::nLayers*sizeof(GPlexHS), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_maxSize, sizeof(int));  //  global maximum
  cudaCheckError();
}

template <typename T>
void FitterCU<T>::freeDevice() {
  d_par_iC.free();
  d_inChg.free();
  d_par_iP.free();
  d_errorProp.free();
  d_Err_iP.free();
  d_Err_iC.free();
  for (int hi = 0; hi < Config::nLayers; ++hi) {
    d_msPar[hi].free();
    d_msErr[hi].free();
  }
  cudaFree(d_msPar_arr);
  cudaFree(d_msErr_arr);
  cudaFree(d_maxSize);
  cudaCheckError();
}

template <typename T>
void FitterCU<T>::kalmanUpdateMerged(const int hit_idx) {
  kalmanUpdate_wrapper(stream, d_Err_iP, d_msErr[hit_idx],
                       d_par_iP, d_msPar[hit_idx], d_par_iC, d_Err_iC, N);
}

template <typename T>
void FitterCU<T>::kalmanUpdate_standalone(
    const MPlexLS &psErr, const MPlexLV& psPar, const MPlexQI &inChg,
    const MPlexHS &msErr, const MPlexHV& msPar,
    MPlexLS &outErr, MPlexLV& outPar,
    const int hit_idx, const int N_proc)
{
  d_Err_iP.copyAsyncFromHost(stream, psErr);
  d_msErr[hit_idx].copyAsyncFromHost(stream, msErr);
  d_par_iP.copyAsyncFromHost(stream, psPar);
  d_msPar[hit_idx].copyAsyncFromHost(stream, msPar);

  kalmanUpdate_wrapper(stream, d_Err_iP, d_msErr[hit_idx],
                       d_par_iP, d_msPar[hit_idx], d_par_iC, d_Err_iC, N_proc);

  d_par_iC.copyAsyncToHost(stream, outPar);
  d_Err_iC.copyAsyncToHost(stream, outErr);
}

template <typename T>
void FitterCU<T>::propagationMerged(const int hit_idx) {
  propagation_wrapper(stream, d_msPar[hit_idx], d_Err_iC, d_par_iC, d_inChg,
                      d_par_iP, d_Err_iP, false, N);
}


// FIXME: Temporary. Separate allocations / transfers
template <typename T>
void FitterCU<T>::allocate_extra_addBestHit() {
  d_XHitPos.allocate(Nalloc);
  d_XHitSize.allocate(Nalloc);
  d_XHitArr.allocate(Nalloc);
  cudaMalloc((void**)&d_HitsIdx_arr, Config::nLayers * sizeof(GPlexQI));
  for (int hi = 0; hi < Config::nLayers; ++hi) {
    d_HitsIdx[hi].allocate(Nalloc);
  }
  cudaMemcpy(d_HitsIdx_arr, d_HitsIdx, Config::nLayers*sizeof(GPlexQI), cudaMemcpyHostToDevice);
  d_Chi2.allocate(Nalloc);
  d_Label.allocate(Nalloc);
  cudaCheckError();
}

template <typename T>
void FitterCU<T>::free_extra_addBestHit() {
  for (int hi = 0; hi < Config::nLayers; ++hi) {
    d_HitsIdx[hi].free(); cudaCheckError();
  }
  cudaFree(d_HitsIdx_arr);
  d_Label.free(); cudaCheckError();
  d_Chi2.free(); cudaCheckError();

  d_XHitArr.free(); cudaCheckError();
  d_XHitSize.free(); cudaCheckError();
  d_XHitPos.free(); cudaCheckError();
}

template <typename T>
void FitterCU<T>::setHitsIdxToZero(const int hit_idx) {
  cudaMemset(d_HitsIdx[hit_idx].ptr, 0, Nalloc*sizeof(int));
}

template <typename T>
void FitterCU<T>::InputTracksAndHitIdx(const EtaBinOfCandidatesCU &etaBin,
                              const int beg, const int end, const bool inputProp) {
  InputTracksCU_wrapper(stream, etaBin, d_Err_iP, d_par_iP,
                        d_inChg, d_Chi2, d_Label, d_HitsIdx_arr,
                        beg, end, inputProp, N);
}

template <typename T>
void FitterCU<T>::OutputTracksAndHitIdx(EtaBinOfCandidatesCU &etaBin,
                               const int beg, const int end, const bool outputProp) {
  OutputTracksCU_wrapper(stream, etaBin, d_Err_iP, d_par_iP,
                         d_inChg, d_Chi2, d_Label, d_HitsIdx_arr,
                         beg, end, outputProp, N);
  cudaStreamSynchronize(stream);
  cudaCheckError();
}


template <typename T>
void FitterCU<T>::InputTracksAndHitIdxComb(const EtaBinOfCombCandidatesCU &etaBin,
      const int Nhits, const int beg, const int end, const bool inputProp) {
  InputTracksAndHitIdxComb_wrapper(stream, 
                                   etaBin,
                                   d_Err_iP, d_par_iP,
                                   d_inChg, d_Chi2,
                                   d_Label, d_HitsIdx,
                                   d_SeedIdx, d_CandIdx,
                                   d_Valid, Nhits,
                                   beg, end, 
                                   inputProp, N);
}


template <typename T>
void FitterCU<T>::propagateTracksToR(const float radius, const int N) {
  propagationForBuilding_wrapper(stream, d_Err_iC, d_par_iC, d_inChg, 
                                 radius, d_Err_iP, d_par_iP, N); 
}

template <typename T>
void FitterCU<T>::propagateTracksToR_standalone(const float radius, const int N,
    const MPlexLS& Err_iC, const MPlexLV& Par_iC, const MPlexQI& inChg, 
    MPlexLS& Err_iP, MPlexLV& Par_iP) {
  d_Err_iC.copyAsyncFromHost(stream, Err_iC);
  d_par_iC.copyAsyncFromHost(stream, Par_iC);
  d_inChg.copyAsyncFromHost(stream, inChg);

  propagateTracksToR(radius, N);

  d_Err_iP.copyAsyncToHost(stream, Err_iP);
  d_par_iP.copyAsyncToHost(stream, Par_iP);
  cudaStreamSynchronize(stream);
}

template <typename T>
void FitterCU<T>::FitTracks(Track *tracks_cu, int num_tracks,
                            EventOfHitsCU &events_of_hits_cu,
                            int Nhits, const bool useParamBfield)
{
  float etime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
 
  for (int itrack = 0; itrack < num_tracks; itrack += Nalloc)
  {
    int beg = itrack;
    int end = std::min(itrack + Nalloc, num_tracks);
    setNumberTracks(end-beg);

    InputTracksAndHitsCU_wrapper(stream, tracks_cu, events_of_hits_cu,
                                 d_Err_iC, d_par_iC,
                                 d_msErr_arr, d_msPar_arr,
                                 d_inChg, d_Chi2, d_Label,
                                 d_HitsIdx_arr, beg, end, false, N);
    fittracks_wrapper(stream, d_Err_iP, d_par_iP, d_msErr_arr, d_msPar_arr,
                      d_Err_iC, d_par_iC, d_errorProp, d_inChg, useParamBfield,
                      Nhits, N);
    OutputFittedTracksCU_wrapper(stream, tracks_cu, 
                                 d_Err_iC, d_par_iC,
                                 d_inChg, d_Chi2, d_Label,
                                 beg, end, N);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&etime, start, stop);
  std::cerr << "CUDA etime: " << etime << " ms.\n";
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
