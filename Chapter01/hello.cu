#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU! %d\n", threadIdx.x);
}

int main() {
    printf("Hello world from CPU!\n");
    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}