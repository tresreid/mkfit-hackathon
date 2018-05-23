objects = gplex_mul.o  gpu_utils.o

multorture: $(objects) multorture.cc
	nvcc -arch=sm_52 -o multorture multorture.cc $(objects)

%.o: %.cu
	nvcc -arch=sm_52 -I ${CUB_ROOT}/include -c $< -o $@  

clean:
	rm -f *.o multorture