C_APPS=hello

all: ${C_APPS}

%:%.cu
	nvcc $< -o $@.o
clean:
	rm -f $(addsuffix .o, ${C_APPS})