#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This code implements the interleaved and neighbor-paired approaches to
 * parallel reduction in CUDA. For this example, the sum operation is used. A
 * variety of optimizations on parallel reduction aimed at reducing divergence
 * are also demonstrated, such as unrolling.
 */

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size) {
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {

}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata,
                                     unsigned int n) {

}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) {
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n) {
}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n) {

}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n) {

}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n) {

}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata,
                                           unsigned int n) {
}

template<unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata,
                                     unsigned int n) {
}

__global__ void reduceUnrollWarps(int *g_idata, int *g_odata, unsigned int n) {
}

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if (argc > 1) {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int) (rand() & 0xFF);
    }

    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

//    // kernel 2: reduceNeighbored with less divergence
//    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//    CHECK(cudaDeviceSynchronize());
//    iStart = seconds();
//    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
//    CHECK(cudaDeviceSynchronize());
//    iElaps = seconds() - iStart;
//    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
//                     cudaMemcpyDeviceToHost));
//    gpu_sum = 0;
//
//    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
//
//    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
//           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);
//
//    // kernel 3: reduceInterleaved
//    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//    CHECK(cudaDeviceSynchronize());
//    iStart = seconds();
//    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
//    CHECK(cudaDeviceSynchronize());
//    iElaps = seconds() - iStart;
//    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
//                     cudaMemcpyDeviceToHost));
//    gpu_sum = 0;
//
//    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
//
//    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
//           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);
//
//    // kernel 4: reduceUnrolling2
//    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//    CHECK(cudaDeviceSynchronize());
//    iStart = seconds();
//    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
//    CHECK(cudaDeviceSynchronize());
//    iElaps = seconds() - iStart;
//    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),
//                     cudaMemcpyDeviceToHost));
//    gpu_sum = 0;
//
//    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];
//
//    printf("gpu Unrolling2  elapsed %f sec gpu_sum: %d <<<grid %d block "
//           "%d>>>\n", iElaps, gpu_sum, grid.x / 2, block.x);
//
//    // kernel 5: reduceUnrolling4
//    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//    CHECK(cudaDeviceSynchronize());
//    iStart = seconds();
//    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
//    CHECK(cudaDeviceSynchronize());
//    iElaps = seconds() - iStart;
//    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
//                     cudaMemcpyDeviceToHost));
//    gpu_sum = 0;
//
//    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];
//
//    printf("gpu Unrolling4  elapsed %f sec gpu_sum: %d <<<grid %d block "
//           "%d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);
//
//    // kernel 6: reduceUnrolling8
//    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//    CHECK(cudaDeviceSynchronize());
//    iStart = seconds();
//    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
//    CHECK(cudaDeviceSynchronize());
//    iElaps = seconds() - iStart;
//    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
//                     cudaMemcpyDeviceToHost));
//    gpu_sum = 0;
//
//    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
//
//    printf("gpu Unrolling8  elapsed %f sec gpu_sum: %d <<<grid %d block "
//           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);
//
//    for (int i = 0; i < grid.x / 16; i++) gpu_sum += h_odata[i];
//
//    // kernel 8: reduceUnrollWarps8
//    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//    CHECK(cudaDeviceSynchronize());
//    iStart = seconds();
//    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
//    CHECK(cudaDeviceSynchronize());
//    iElaps = seconds() - iStart;
//    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
//                     cudaMemcpyDeviceToHost));
//    gpu_sum = 0;
//
//    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
//
//    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block "
//           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);
//
//
//    // kernel 9: reduceCompleteUnrollWarsp8
//    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//    CHECK(cudaDeviceSynchronize());
//    iStart = seconds();
//    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
//    CHECK(cudaDeviceSynchronize());
//    iElaps = seconds() - iStart;
//    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
//                     cudaMemcpyDeviceToHost));
//    gpu_sum = 0;
//
//    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
//
//    printf("gpu Cmptnroll8  elapsed %f sec gpu_sum: %d <<<grid %d block "
//           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);
//
//    // kernel 9: reduceCompleteUnroll
//    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
//    CHECK(cudaDeviceSynchronize());
//    iStart = seconds();
//
//    switch (blocksize) {
//        case 1024:
//            reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata,
//                                                              size);
//            break;
//
//        case 512:
//            reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,
//                                                             size);
//            break;
//
//        case 256:
//            reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata,
//                                                             size);
//            break;
//
//        case 128:
//            reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata,
//                                                             size);
//            break;
//
//        case 64:
//            reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
//            break;
//    }

//    CHECK(cudaDeviceSynchronize());
//    iElaps = seconds() - iStart;
//    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
//                     cudaMemcpyDeviceToHost));
//
//    gpu_sum = 0;
//
//    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
//
//    printf("gpu Cmptnroll   elapsed %f sec gpu_sum: %d <<<grid %d block "
//           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if (!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}