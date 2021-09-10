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
__global__ void warmup(int *g_idata, int *g_odata, unsigned int n) {
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


__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n) {
    int bidx = 8 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    int *i_data = g_idata + bidx;
    if (bidx + tid + 7 * blockDim.x < n) {
        int a1 = i_data[tid + blockDim.x];
        int a2 = i_data[tid + 2 * blockDim.x];
        int a3 = i_data[tid + 3 * blockDim.x];
        int b1 = i_data[tid + 4 * blockDim.x];
        int b2 = i_data[tid + 5 * blockDim.x];
        int b3 = i_data[tid + 6 * blockDim.x];
        int b4 = i_data[tid + 7 * blockDim.x];
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

__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n) {
    int bidx = 16 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    int *i_data = g_idata + bidx;
    if (bidx + tid + 15 * blockDim.x < n) {
        int a1 = i_data[tid + blockDim.x];
        int a2 = i_data[tid + 2 * blockDim.x];
        int a3 = i_data[tid + 3 * blockDim.x];
        int b1 = i_data[tid + 4 * blockDim.x];
        int b2 = i_data[tid + 5 * blockDim.x];
        int b3 = i_data[tid + 6 * blockDim.x];
        int b4 = i_data[tid + 7 * blockDim.x];
        int b5 = i_data[tid + 8 * blockDim.x];
        int b6 = i_data[tid + 9 * blockDim.x];
        int b7 = i_data[tid + 10 * blockDim.x];
        int b8 = i_data[tid + 11 * blockDim.x];
        int b9 = i_data[tid + 12 * blockDim.x];
        int b10 = i_data[tid + 13 * blockDim.x];
        int b11 = i_data[tid + 14 * blockDim.x];
        int b12 = i_data[tid + 15 * blockDim.x];
        i_data[tid] += a1 + a2 + a3 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11 + b12;
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

__global__ void reduceUnrolling8ForLoop(int *g_idata, int *g_odata, unsigned int n) {
    int bidx = 8 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    int *i_data = g_idata + bidx;

    if (bidx + tid + 7 * blockDim.x < n) {
        int *ptr = g_idata + bidx + tid;
        int tmp = 0;
        // Increment tmp 8 times with values strided by blockDim.x
        for (int i = 0; i < 8; i++) {
            tmp += *ptr;
            ptr += blockDim.x;
        }
        i_data[tid] = tmp;
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

    // kernel 0: warmup
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 1: reduceUnrolling8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);


    // kernel 2: reduceUnrolling8ForLoop
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling8ForLoop<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrolling8ForLoop  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 3: reduceUnrolling16
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling16<<<grid.x / 16, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 16 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 16; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling16  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 16, block.x);

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