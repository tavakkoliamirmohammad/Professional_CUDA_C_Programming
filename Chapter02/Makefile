C_APPS=sumArrayOnHost

CU_APPS=checkDimension sumArraysOnGPU-small-case sumArraysOnGPU-timer sumMatrixOnGPU-2D-grid-2D-block \
 sumMatrixOnGPU-2D-grid-1D-block-1D checkDeviceInfo

NVCC_FLAGS= -O3 -arch=$(CUDA_ARCH)

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc ${NVCC_FLAGS} -o $@.o $<
%: %.c
	gcc -std=c99 -o $@.o $<
clean:
	rm -f $(addsuffix .o, ${CU_APPS}) $(addsuffix .o, ${C_APPS})