include Makefile.config

TGTS := main

EXES := ${TGTS}

.PHONY: all clean distclean

all: ${EXES}

ifdef USE_CUDA
CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:.cu=.o)

LDFLAGS_CU := -lcudart
LDFLAGS_CU += -lnvToolsExt

# To share __device__ function across several source files
# 1) compile with --device-c (works as -c should be)
# 2) Create some kind of dictionary with all the .cu.o
# 3) The dictionary AND the original .o files should be used
# -- Works only for CUDA_VERSION >= 5
# TODO: Clean the "-I.. -std=c++11"
${CU_OBJS}: %.o: %.cu
	${NV} --device-c -o $@ $< -I. -std=c++11 -DUSE_MATRIPLEX \
		-Itrove ${NVCCFLAGS} -Wno-deprecated-declarations \
		-lineinfo -g --ptxas-options="-v -flcm=ca"
# --keep --keep-dir ptxs
# -Wno-deprecated-gpu-targets \
# --use_fast_math
# -g -G

CU_LINK := cu_link.o
${CU_LINK}: ${CU_OBJS}
	${NV} --device-link $^ --output-file $@
endif

SRCS := $(wildcard *.cc)
OBJS := $(SRCS:.cc=.o)
DEPS := $(SRCS:.cc=.d)

ALLOBJS := ${OBJS} ${CU_OBJS} ${CU_LINK}

ifeq ($(filter clean-local clean distclean, ${MAKECMDGOALS}),)
include ${DEPS}
endif

clean-local:
	-rm -f ${EXES} *.d *.o *.om *.so
	-rm -rf main.dSYM

clean: clean-local
	cd mkFit && ${MAKE} clean

distclean: clean-local
	-rm -f *.optrpt

main: ${AUTO_TGTS} ${ALLOBJS}
	${CXX} ${CXXFLAGS} ${VEC_HOST} -o $@ ${ALLOBJS} ${LDFLAGS} ${LDFLAGS_HOST} ${LDFLAGS_CU}

${OBJS}: %.o: %.cc %.d
	${CXX} ${CPPFLAGS} ${CXXFLAGS} ${VEC_HOST} -c -o $@ $<

echo:
	-echo CXX = ${CXX}
