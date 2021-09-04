C_APPS=
CU_APPS=simpleDivergence sumMatrix

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O3 -arch=sm_60 -o $@.o $<
%: %.c
	gcc -std=c99 -o $@.o $<
clean:
	rm -f $(addsuffix .o, ${CU_APPS}) $(addsuffix .o, ${C_APPS})