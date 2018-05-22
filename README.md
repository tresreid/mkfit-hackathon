# mkfit-hackathon
mkfit session for hackathon

To start, get cub and dependencies
source /cvmfs/cms.cern.ch/slc6_amd64_gcc700/external/cub/1.8.0/etc/profile.d/dependencies-setup.sh
source /cvmfs/cms.cern.ch/slc6_amd64_gcc700/external/cub/1.8.0/etc/profile.d/init.sh

compile

```
nvcc -c gplex_mul.cu  -I $CUB_ROOT/include
nvcc -o multorture multorture.cc gplex_mul.o
```