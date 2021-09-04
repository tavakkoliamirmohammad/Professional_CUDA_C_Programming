C_APPS=sumArrayOnHost
CU_APPS=checkDimension sumArraysOnGPU-small-case sumArraysOnGPU-timer sumMatrixOnGPU-2D-grid-2D-block \
 sumMatrixOnGPU-2D-grid-1D-block-1D

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -o $@.o $<
%: %.c
	gcc -std=c99 -o $@.o $<
clean:
	rm -f ${CU_APPS} ${C_APPS}