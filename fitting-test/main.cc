/*
  g++ -std=c++11 -O3 -Wall -Wno-unknown-pragmas -o main main.cc Track.cc Hit.cc Matrix.cc KalmanUtils.cc Propagation.cc Simulation.cc ConformalUtils.cc buildtest.cc fittest.cc -I. `root-config --libs --cflags`
  icc -std=gnu++0x -O3 -openmp -o main main.cc Track.cc Hit.cc Matrix.cc KalmanUtils.cc Propagation.cc Simulation.cc buildtest.cc fittest.cc -I. `root-config --libs --cflags`
*/

#include <sstream>
#include <chrono>
#include <list>
#include <memory>

#include "Matrix.h"
#include "Event.h"
#include "Validation.h"

#ifdef TBB
#include "tbb/task_scheduler_init.h"
#endif

//#define CYLINDER

void initGeom(Geometry& geom)
{
  std::cout << "Constructing SimpleGeometry Cylinder geometry" << std::endl;
  // NB: we currently assume that each node is a layer, and that layers
  // are added starting from the center
  // NB: z is just a dummy variable, VUSolid is actually infinite in size.  *** Therefore, set it to the eta of simulation ***

  for (int l = 0; l < Config::nLayers; l++) {
    float r = (l+1)*Config::fRadialSpacing;
    float z = r / std::tan(2.0*std::atan(std::exp(-Config::fEtaDet))); // calculate z extent based on eta, r
    VUSolid* utub = new VUSolid(r, r+Config::fRadialExtent);
    geom.AddLayer(utub, r, z);
  }
}

typedef std::chrono::time_point<std::chrono::system_clock> timepoint;
typedef std::chrono::duration<double> tick;

static timepoint now()
{
  return std::chrono::system_clock::now();
}

static tick delta(timepoint& t0)
{
  timepoint t1(now());
  tick d = t1 - t0;
  t0 = t1;
  return d;
}

// also from mkfit
typedef std::list<std::string> lStr_t;
typedef lStr_t::iterator       lStr_i;

void next_arg_or_die(lStr_t& args, lStr_i& i, bool allow_single_minus=false)
{
  lStr_i j = i;
  if (++j == args.end() ||
      ((*j)[0] == '-' && ! (*j == "-" && allow_single_minus)))
  {
    std::cerr <<"Error: option "<< *i <<" requires an argument.\n";
    exit(1);
  }
  i = j;
}

int main(int argc, const char* argv[])
{

  auto nThread = 1;

  // following mkFit on argv
  lStr_t mArgs; 
  for (int i = 1; i < argc; ++i)
  {
    mArgs.push_back(argv[i]);
  }

  lStr_i i  = mArgs.begin();
  while (i != mArgs.end())
  {
    lStr_i start = i;

    if (*i == "-h" || *i == "-help" || *i == "--help")
    {
      printf(
        "Usage: %s [options]\n"
        "Options:\n"
        "  --num-events    <num>    number of events to run over (def: %d)\n"
        "  --num-tracks    <num>    number of tracks to generate for each event (def: %d)\n"
        "  --num-thr       <num>    number of threads used for TBB  (def: %d)\n"
        ,
        argv[0],
        Config::nEvents,
        Config::nTracks,
        nThread
      );
      exit(0);
    }
    else if (*i == "--num-events")
    {
      next_arg_or_die(mArgs, i);
      Config::nEvents = atoi(i->c_str());
    }
    else if (*i == "--num-tracks")
    {
      next_arg_or_die(mArgs, i);
      Config::nTracks = atoi(i->c_str());
    }
    else if (*i == "--num-thr")
    {
      next_arg_or_die(mArgs, i);
      nThread = atoi(i->c_str());
    }
   else
    {
      fprintf(stderr, "Error: Unknown option/argument '%s'.\n", i->c_str());
      exit(1);
    }
    mArgs.erase(start, ++i);
  }

  Geometry geom;
  initGeom(geom);
  std::unique_ptr<Validation> val(Validation::make_validation("valtree.root"));

  for ( int i = 0; i < Config::nLayers; ++i ) {
    std::cout << "Layer = " << i << ", Radius = " << geom.Radius(i) << std::endl;
  }

#ifdef TBB
  std::cout << "Initializing with " << nThread << " threads." << std::endl;
  tbb::task_scheduler_init tasks(nThread);
#endif

  for (int evt=0; evt<Config::nEvents; ++evt) {
    Event ev(geom, *val, evt, nThread);
    std::cout << "EVENT #"<< ev.evtID() << std::endl;

    ev.Simulate();
    ev.Segment();
    ev.Fit();
  }

  return 0;
}
