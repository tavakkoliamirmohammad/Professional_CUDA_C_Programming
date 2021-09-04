C_APPS=sumArrayOnHost
CU_APPS=checkDimension

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -arch=sm_60 -o $@.o $<
%: %.c
	gcc -std=c99 -o $@.o $<
clean:
	rm -f ${CU_APPS} ${C_APPS}