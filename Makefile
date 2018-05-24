#eigenopt = -Ieigen -I$(shell cd $$CMSSW_BASE && scram tool tag cuda INCLUDE)/crt --expt-relaxed-constexpr
objects = gplex_mul.o  gpu_utils.o multorture.o
#arch = --gpu-architecture=compute_35
arch = -arch=sm_60
nvccflags = $(arch) $(cudainc)

multorture: $(objects)
	nvcc $(nvccflags) -o $@ $(objects)

multorture.o: multorture.cc gplex_mul.h
gplex_mul.o: gplex_mul.cu gpu_utils.h GPlex.h
gpu_utils.o: gpu_utils.cu gpu_utils.h

%.o: %.cu
	nvcc $(nvccflags) -I${CUB_ROOT}/include -c $< -o $@  

%.o: %.cc
	nvcc $(nvccflags) -I${CUB_ROOT}/include -c $< -o $@  

clean:
	rm -f *.o multorture
