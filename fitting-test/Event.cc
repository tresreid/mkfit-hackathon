#include "Event.h"
#include "Simulation.h"
#include "BinInfoUtils.h"

//#define DEBUG
#include "Debug.h"

#ifdef TBB
#include "tbb/tbb.h"
#endif

std::mutex Event::printmutex;

inline bool sortByPhi(const Hit& hit1, const Hit& hit2)
{
  return hit1.phi()<hit2.phi();
}

static bool tracksByPhi(const Track& t1, const Track& t2)
{
  return t1.posPhi()<t2.posPhi();
}

void Event::resetLayerHitMap(bool resetSimHits) {
  //gc: not sure what is being done here
  layerHitMap_.clear();
  layerHitMap_.resize(simHitsInfo_.size());
  for (int ilayer = 0; ilayer < layerHits_.size(); ++ilayer) {
    for (int index = 0; index < layerHits_[ilayer].size(); ++index) {
      auto& hit = layerHits_[ilayer][index];
      assert(hit.mcHitID() >= 0); // tmp debug
      assert(hit.mcHitID() < layerHitMap_.size());
      layerHitMap_[hit.mcHitID()] = HitID(ilayer, index);
    }
  }
  if (resetSimHits) {
    for (auto&& track : simTracks_) {
      for (int il = 0; il < track.nTotalHits(); ++il) {
        assert(layerHitMap_[track.getHitIdx(il)].index >= 0); // tmp debug
        track.setHitIdx(il, layerHitMap_[track.getHitIdx(il)].index);
      }
    }
  }
}

Event::Event(const Geometry& g, Validation& v, int evtID, int threads) : geom_(g), validation_(v), evtID_(evtID), threads_(threads), mcHitIDCounter_(0)
{
  layerHits_.resize(Config::nLayers);
  segmentMap_.resize(Config::nLayers);
  cuFitters_.resize(threads);
  cuBuilders_.resize(threads);
  for (int i = 0; i < threads; ++i) {
    constexpr int gplex_width = 20000;

    cuFitters_[i].reset(new FitterCU<float>(gplex_width));
    auto fitter = cuFitters_[i].get();

    fitter->allocateDevice();
    fitter->allocate_extra_addBestHit();
    fitter->allocate_extra_combinatorial();
    fitter->createStream();
    fitter->setNumberTracks(gplex_width);

    cuBuilders_[i].reset(new BuilderCU(fitter));
    auto builder = cuBuilders_[i].get();

    builder->allocateGeometry(geom_);
  }
}

Event::~Event()
{
  for (auto& fp : cuFitters_) {
    auto fitter = fp.get();
    fitter->freeDevice();
    fitter->free_extra_addBestHit();
    fitter->free_extra_combinatorial();
    fitter->destroyStream();
  }
}

void Event::Reset(int evtID)
{
  evtID_ = evtID;
  mcHitIDCounter_ = 0;

  for (auto&& l : layerHits_) { l.clear(); }
  for (auto&& l : segmentMap_) { l.clear(); }

  simHitsInfo_.clear();
  layerHitMap_.clear();
  simTracks_.clear();
  seedTracks_.clear();
  candidateTracks_.clear();
  fitTracks_.clear();
  simTrackStates_.clear();

  validation_.resetValidationMaps(); // need to reset maps for every event.
  if (Config::super_debug) {
    validation_.resetDebugVectors(); // need to reset vectors for every event.
  }
}

void Event::Simulate()
{
  simTracks_.resize(Config::nTracks);
  simHitsInfo_.resize(Config::nTotHit * Config::nTracks);
  for (auto&& l : layerHits_) {
    l.resize(Config::nTracks);  // thread safety
  }
  simTrackStates_.resize(Config::nTracks);

#ifdef TBB
  parallel_for( tbb::blocked_range<size_t>(0, Config::nTracks, 100), 
      [&](const tbb::blocked_range<size_t>& itracks) {

    const Geometry tmpgeom(geom_.clone()); // USolids isn't thread safe??
    for (auto itrack = itracks.begin(); itrack != itracks.end(); ++itrack) {
#else
    const Geometry& tmpgeom(geom_);
    for (int itrack=0; itrack<Config::nTracks; ++itrack) {
#endif
      //create the simulated track
      SVector3 pos;
      SVector3 mom;
      SMatrixSym66 covtrk;
      HitVec hits;
      TSVec  initialTSs;
      // int starting_layer  = 0; --> for displaced tracks, may want to consider running a separate Simulate() block with extra parameters

      int q=0;//set it in setup function
      // do the simulation
      if (Config::useCMSGeom) setupTrackFromTextFile(pos,mom,covtrk,hits,*this,itrack,q,tmpgeom,initialTSs);
      else if (Config::endcapTest) setupTrackByToyMCEndcap(pos,mom,covtrk,hits,*this,itrack,q,tmpgeom,initialTSs);
      else setupTrackByToyMC(pos,mom,covtrk,hits,*this,itrack,q,tmpgeom,initialTSs); 

      // uber ugly way of getting around read-in / write-out of objects needed for validation
      if (Config::normal_val || Config::fit_val) {simTrackStates_[itrack] = initialTSs;}
      validation_.collectSimTkTSVecMapInfo(itrack,initialTSs); // save initial TS parameters in validation object ... just a copy of the above line

      simTracks_[itrack] = Track(q,pos,mom,covtrk,0.0f);
      auto& sim_track = simTracks_[itrack];
      sim_track.setLabel(itrack);
      for (int ilay = 0; ilay < hits.size(); ++ilay) {
        sim_track.addHitIdx(hits[ilay].mcHitID(),0.0f); // set to the correct hit index after sorting
        layerHits_[ilay][itrack] = hits[ilay];  // thread safety
      }
    }
#ifdef TBB
  });
#endif
}

void Event::Segment()
{
  //sort in phi and dump hits per layer, fill phi partitioning
  for (int ilayer=0; ilayer<layerHits_.size(); ++ilayer) {
    dprint("Hits in layer=" << ilayer);
    
    segmentMap_[ilayer].resize(1);    // only one eta bin for special case, avoid ifdefs
    std::sort(layerHits_[ilayer].begin(), layerHits_[ilayer].end(), sortByPhi);
    std::vector<int> lay_phi_bin_count(Config::nPhiPart);//should it be 63? - yes!
    for (int ihit=0;ihit<layerHits_[ilayer].size();++ihit) {
      dprint("hit r/phi/eta : " << layerHits_[ilayer][ihit].r() << " "
                                << layerHits_[ilayer][ihit].phi() << " " << layerHits_[ilayer][ihit].eta());

      int phibin = getPhiPartition(layerHits_[ilayer][ihit].phi());
      lay_phi_bin_count[phibin]++;
    }

    //now set index and size in partitioning map
    int lastIdxFound = -1;
    for (int bin=0; bin<Config::nPhiPart; ++bin) {
      int binSize = lay_phi_bin_count[bin];
      int firstBinIdx = lastIdxFound+1;
      BinInfo binInfo(firstBinIdx, binSize);
      segmentMap_[ilayer][0].push_back(binInfo); // [0] bin is just the only eta bin ... reduce ifdefs
      if (binSize>0){
        lastIdxFound+=binSize;
      }
    }
  } // end loop over layers

  resetLayerHitMap(true);
}

void Event::Find()
{
}

void Event::Fit()
{
  fitTracks_.resize(simTracks_.size());
  fitTracksExtra_.resize(simTracks_.size());

  Track *tracks_cu;
  auto& cuFitter = *cuFitters_[0].get();

  cudaMalloc((void**)&tracks_cu, simTracks_.size()*sizeof(Track));
  cudaMemcpyAsync(tracks_cu, &simTracks_[0], simTracks_.size()*sizeof(Track),
                  cudaMemcpyHostToDevice, cuFitter.get_stream());

  EventOfHitsCU events_of_hits_cu;
  events_of_hits_cu.reserve_layers(layerHits_);
  events_of_hits_cu.copyFromCPU(layerHits_, cuFitter.get_stream());

  double time = dtime();

  cuFitter.FitTracks(tracks_cu, simTracks_.size(), events_of_hits_cu, Config::nLayers, true);

  cudaMemcpy(&fitTracks_[0], tracks_cu, simTracks_.size()*sizeof(Track), cudaMemcpyDeviceToHost);

  time = dtime() - time;

  cudaFree(tracks_cu);
}

void Event::PrintStats(const TrackVec& trks, TrackExtraVec& trkextras)
{
  int miss(0), found(0), fp_10(0), fp_20(0), hit8(0), h8_10(0), h8_20(0);

  for (auto&& trk : trks) {
    auto&& extra = trkextras[trk.label()];
    extra.setMCTrackIDInfo(trk, layerHits_, simHitsInfo_);
    if (extra.mcTrackID() < 0) {
      ++miss;
    } else {
      auto&& mctrk = simTracks_[extra.mcTrackID()];
      auto pr = trk.pT()/mctrk.pT();
      found++;
      bool h8 = trk.nFoundHits() >= 8;
      bool pt10 = pr > 0.9 && pr < 1.1;
      bool pt20 = pr > 0.8 && pr < 1.2;
      fp_10 += pt10;
      fp_20 += pt20;
      hit8 += h8;
      h8_10 += h8 && pt10;
      h8_20 += h8 && pt20;
    }
  }
  std::cout << "found tracks=" << found   << "  in pT 10%=" << fp_10    << "  in pT 20%=" << fp_20    << "     no_mc_assoc="<< miss <<std::endl
            << "  nH >= 8   =" << hit8    << "  in pT 10%=" << h8_10    << "  in pT 20%=" << h8_20    << std::endl;
}
