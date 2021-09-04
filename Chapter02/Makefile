C_APPS=sumArrayOnHost
CU_APPS=checkDimension sumArraysOnGPU-small-case sumArraysOnGPU-timer sumMatrixOnGPU-2D-grid-2D-block \
 sumMatrixOnGPU-2D-grid-1D-block-1D checkDeviceInfo

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -o $@.o $<
%: %.c
	gcc -std=c99 -o $@.o $<
clean:
	rm -f $(addsuffix .o, ${CU_APPS}) $(addsuffix .o, ${C_APPS})