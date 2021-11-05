C_APPS=
CU_APPS= simpleHyperQDepthFirst simpleHyperQBreadthFirst simpleHyperQOpenMP simpleHyperQDependency \
	simpleVecAddDepthFirst simpleVecAddBreadthFirst AsyncAPI simpleCallback
NVCC_FLAGS= -O3 -Wno-deprecated-gpu-targets  -Xcompiler -fopenmp
all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc ${NVCC_FLAGS} -o $@.o $<
%: %.c
	gcc -std=c99 -o $@.o $<
clean:
	rm -f $(addsuffix .o, ${CU_APPS}) $(addsuffix .o, ${C_APPS})