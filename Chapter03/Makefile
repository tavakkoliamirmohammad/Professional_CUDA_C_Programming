C_APPS=
CU_APPS=simpleDivergence sumMatrix reduceInteger reduceIntegerExercise reduceIntegerSync
NVCC_FLAGS= -O3 -arch=sm_60
all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc ${NVCC_FLAGS} -o $@.o $<
%: %.c
	gcc -std=c99 -o $@.o $<
clean:
	rm -f $(addsuffix .o, ${CU_APPS}) $(addsuffix .o, ${C_APPS})