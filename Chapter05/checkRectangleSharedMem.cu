#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using shared memory to transpose rectangular thread coordinates
 * of a CUDA grid into a global memory array. Different kernels below
 * demonstrate performing reads and writes with different ordering, as well as
 * optimizing using memory padding.
 */

#define BDIMX 32
#define BDIMY 16
#define IPAD  2

void printData(const char *msg, int *in,  const int size)
{
    printf("%s: ", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%4d", in[i]);
        fflush(stdout);
    }

    printf("\n\n");
}

__global__ void setRowReadRow(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out)
{
    __shared__ int tile[BDIMX][BDIMY];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadCol2(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int rowIdx = idx % blockDim.y;
    unsigned int colIdx = idx / blockDim.y;

    tile[rowIdx][colIdx] = idx;
    __syncthreads();

    out[idx] = tile[rowIdx][colIdx];
}

__global__ void setRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int rowIdx = idx % blockDim.y;
    unsigned int colIdx = idx / blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

    out[idx] = tile[rowIdx][colIdx];
}

__global__ void setRowReadColPad(int *out)
{

}

__global__ void setRowReadColDyn(int *out)
{
    extern __shared__ int tile[];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int rowIdx = idx % blockDim.y;
    unsigned int colIdx = idx / blockDim.y;

    unsigned int tileIdx = rowIdx * blockDim.x + colIdx;

    tile[idx] = idx;
    __syncthreads();

    out[idx] = tile[tileIdx];
}

__global__ void setRowReadColDynPad(int *out)
{

   
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    CHECK(cudaDeviceGetSharedMemConfig ( &pConfig ));
    printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

    // set up array size
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = 0;

    if (argc > 1) iprintf = atoi(argv[1]);

    size_t nBytes = nx * ny * sizeof(int);

    // execution configuration
    dim3 block (BDIMX, BDIMY);
    dim3 grid  (1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
            block.y);

    // allocate device memory
    int *d_C;
    CHECK(cudaMalloc((int**)&d_C, nBytes));
    int *gpuRef  = (int *)malloc(nBytes);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadRow       ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setColReadCol       ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol2<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setColReadCol2      ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadCol       ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, BDIMX*BDIMY*sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadColDyn    ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadColPad    ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + IPAD)*BDIMY*sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("setRowReadColDynPad ", gpuRef, nx * ny);

    // free host and device memory
    CHECK(cudaFree(d_C));
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}