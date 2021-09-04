FILES=hello

all: ${FILES}

%:%.cu
	nvcc $< -o $@.o