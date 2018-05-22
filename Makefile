objects = gplex_mul.o  gpu_utils.o

multorture: $(objects) multorture.cc
	nvcc -o multorture multorture.cc $(objects)

%.o: %.cu
	nvcc -I ${CUB_ROOT}/include -c $< -o $@  

clean:
	rm -f *.o multorture