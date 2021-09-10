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
float recursiveReduce(float *data, int const size) {
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
__global__ void warmup(float *g_idata, float *g_odata, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx >= n) return;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            g_idata[idx] = g_idata[idx] + g_idata[idx + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = g_idata[idx];
    }
}

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored(float *g_idata, float *g_odata, unsigned int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx >= n) return;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            g_idata[idx] = g_idata[idx] + g_idata[idx + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = g_idata[idx];
    }
}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess(float *g_idata, float *g_odata,
                                     unsigned int n) {
    int bidx = blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    if (bidx + tid >= n) return;
    float *i_data = g_idata + bidx;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = 2 * stride * tid;
        if (idx + stride < blockDim.x) {
            i_data[idx] = i_data[idx] + i_data[idx + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved(float *g_idata, float *g_odata, unsigned int n) {
    int bidx = blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    if (bidx + tid >= n) return;
    float *i_data = g_idata + bidx;
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            i_data[tid] += i_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

__global__ void reduceUnrolling2(float *g_idata, float *g_odata, unsigned int n) {
    int bidx = 2 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *i_data = g_idata + bidx;
    if (bidx + tid + blockDim.x < n)
        i_data[tid] += i_data[tid + blockDim.x];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            i_data[tid] += i_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

__global__ void reduceUnrolling4(float *g_idata, float *g_odata, unsigned int n) {
    int bidx = 4 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *i_data = g_idata + bidx;
    if (bidx + tid + 3 * blockDim.x < n) {
        float a1 = i_data[tid + blockDim.x];
        float a2 = i_data[tid + 2 * blockDim.x];
        float a3 = i_data[tid + 3 * blockDim.x];
        i_data[tid] += a1 + a2 + a3;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            i_data[tid] += i_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

__global__ void reduceUnrolling8(float *g_idata, float *g_odata, unsigned int n) {
    int bidx = 8 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *i_data = g_idata + bidx;
    if (bidx + tid + 7 * blockDim.x < n) {
        float a1 = i_data[tid + blockDim.x];
        float a2 = i_data[tid + 2 * blockDim.x];
        float a3 = i_data[tid + 3 * blockDim.x];
        float b1 = i_data[tid + 4 * blockDim.x];
        float b2 = i_data[tid + 5 * blockDim.x];
        float b3 = i_data[tid + 6 * blockDim.x];
        float b4 = i_data[tid + 7 * blockDim.x];
        i_data[tid] += a1 + a2 + a3 + b1 + b2 + b3 + b4;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            i_data[tid] += i_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

__global__ void reduceUnrollWarps8(float *g_idata, float *g_odata, unsigned int n) {
    int bidx = 8 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *i_data = g_idata + bidx;
    if (bidx + tid + 7 * blockDim.x < n) {
        float a1 = i_data[tid + blockDim.x];
        float a2 = i_data[tid + 2 * blockDim.x];
        float a3 = i_data[tid + 3 * blockDim.x];
        float b1 = i_data[tid + 4 * blockDim.x];
        float b2 = i_data[tid + 5 * blockDim.x];
        float b3 = i_data[tid + 6 * blockDim.x];
        float b4 = i_data[tid + 7 * blockDim.x];
        i_data[tid] += a1 + a2 + a3 + b1 + b2 + b3 + b4;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            i_data[tid] += i_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile float *vmem = i_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

__global__ void reduceCompleteUnrollWarps8(float *g_idata, float *g_odata,
                                           unsigned int n) {
    int bidx = 8 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *i_data = g_idata + bidx;
    if (bidx + tid + 7 * blockDim.x < n) {
        float a1 = i_data[tid + blockDim.x];
        float a2 = i_data[tid + 2 * blockDim.x];
        float a3 = i_data[tid + 3 * blockDim.x];
        float b1 = i_data[tid + 4 * blockDim.x];
        float b2 = i_data[tid + 5 * blockDim.x];
        float b3 = i_data[tid + 6 * blockDim.x];
        float b4 = i_data[tid + 7 * blockDim.x];
        i_data[tid] += a1 + a2 + a3 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) i_data[tid] += i_data[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) i_data[tid] += i_data[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) i_data[tid] += i_data[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) i_data[tid] += i_data[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile float *vmem = i_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
}

template<unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(float *g_idata, float *g_odata,
                                     unsigned int n) {
    int bidx = 8 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    float *i_data = g_idata + bidx;
    if (bidx + tid + 7 * blockDim.x < n) {
        float a1 = i_data[tid + blockDim.x];
        float a2 = i_data[tid + 2 * blockDim.x];
        float a3 = i_data[tid + 3 * blockDim.x];
        float b1 = i_data[tid + 4 * blockDim.x];
        float b2 = i_data[tid + 5 * blockDim.x];
        float b3 = i_data[tid + 6 * blockDim.x];
        float b4 = i_data[tid + 7 * blockDim.x];
        i_data[tid] += a1 + a2 + a3 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    if (iBlockSize >= 1024 && tid < 512) i_data[tid] += i_data[tid + 512];
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256) i_data[tid] += i_data[tid + 256];
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128) i_data[tid] += i_data[tid + 128];
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64) i_data[tid] += i_data[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile float *vmem = i_data;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = i_data[0];
    }
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
    size_t bytes = size * sizeof(float);
    float *h_idata = (float *) malloc(bytes);
    float *h_odata = (float *) malloc(grid.x * sizeof(float));
    float *tmp = (float *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (float) (rand() & 0xFF);
    }

    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    float gpu_sum = 0;

    // allocate device memory
    float *d_idata = NULL;
    float *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(float)));

    // cpu reduction
    iStart = seconds();
    float cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %f\n", iElaps, cpu_sum);

    // kernel 0: warmup
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling2  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 5: reduceUnrolling4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling4  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 6: reduceUnrolling8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling8  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 8: reduceUnrollWarps8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);


    // kernel 9: reduceCompleteUnrollWarsp8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll8  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 9: reduceCompleteUnroll
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    switch (blocksize) {
        case 1024:
            reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata,
                                                              size);
            break;

        case 512:
            reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,
                                                             size);
            break;

        case 256:
            reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata,
                                                             size);
            break;

        case 128:
            reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata,
                                                             size);
            break;

        case 64:
            reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
    }

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(float),
                     cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll   elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

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