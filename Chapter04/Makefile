C_APPS=
CU_APPS=globalVariable memTransfer pinMemTransfer sumArrayZeroCopy readSegment writeSegment \
		simpleMathAOS simpleMathSOA readSegmentUnroll transpose
NVCC_FLAGS= -O3 -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets 
all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc ${NVCC_FLAGS} -o $@.o $<
%: %.c
	gcc -std=c99 -o $@.o $<
clean:
	rm -f $(addsuffix .o, ${CU_APPS}) $(addsuffix .o, ${C_APPS})