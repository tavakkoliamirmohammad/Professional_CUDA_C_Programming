#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    printf("Hello world from CPU!\n");
    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();

    return 0;
}