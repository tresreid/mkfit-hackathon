#eigenopt = -Ieigen -I$(shell cd $$CMSSW_BASE && scram tool tag cuda INCLUDE)/crt --expt-relaxed-constexpr
eigenopt = -Ieigen --expt-relaxed-constexpr
objects = gplex_mul.o raw_mul.o eigen_mul.o gpu_utils.o multorture.o
arch = --gpu-architecture=compute_35
#arch = -arch=sm_70
nvccflags = $(arch) $(cudainc) $(eigenopt) -O3 --std c++14

multorture: $(objects)
	nvcc $(nvccflags) -o $@ $(objects) -lcublas

multorture.o: multorture.cc gplex_mul.h
gplex_mul.o: gplex_mul.cu gplex_mul.h gpu_utils.h GPlex.h
raw_mul.o: raw_mul.cu gplex_mul.h gpu_utils.h GPlex.h
eigen_mul.o: eigen_mul.cu gplex_mul.h gpu_utils.h GPlex.h
gpu_utils.o: gpu_utils.cu gpu_utils.h


%.o: %.cu
	nvcc $(nvccflags) -I${CUBROOT} -I../eigen_tests/eigen -c $< -o $@  

%.o: %.cc
	nvcc $(nvccflags) -I${CUBROOT} -I../eigen_tests/eigen -c $< -o $@  

clean:
	rm -f *.o multorture
