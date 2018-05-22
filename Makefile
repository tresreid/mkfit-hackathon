objects = gplex_mul.o  gpu_utils.o

all: $(objects)
	nvcc -o multorture multorture.cc $(objects)

%.o: %.cu
	nvcc -I ${CUB_ROOT}/include -c $< -o $@  
clean:
	rm -f *.o multorture