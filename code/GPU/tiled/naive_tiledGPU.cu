#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "naive_tiledGPU.hpp"

#include "../../../utils/chronoGPU.hpp"

using namespace std;
#define ThrPerBlock_y 16
#define ThrPerBlock_x 16
#define NbIteration 1


__device__ __constant__ uint8_t* dev_in_global;



__global__ void kernelNaive_tiledGPU(uint8_t *h_out,const int MapWidth, const int MapHeight , const int outMapWidth, const int outMapHeight)
{
    int TiledDimX = int(MapWidth / outMapWidth);
    int TiledDimY = int(MapHeight / outMapHeight);

    __shared__ int max;

    int location = (blockIdx.y * TiledDimY) * (MapWidth) + blockIdx.x * TiledDimX;
    int indexY = threadIdx.y;
    int indexX = threadIdx.x;  
    int initX = indexX;    
    while (indexY < TiledDimY)
    {
        while (indexX < TiledDimX)
        {
            int point = location + indexY * MapWidth + indexX; 
            atomicMax(&max, int(dev_in_global[point]));
            indexX += blockDim.x;
        }
        indexY += blockDim.y;
        indexX = initX;

    }
    __syncthreads();
    h_out[blockIdx.y * outMapWidth + blockIdx.x] = max;
}

void naive_tiledGPU(uint8_t *h_in, uint8_t *h_out,const int MapWidth, const int MapHeight , const int outMapWidth, const int outMapHeight)
{
    uint8_t *dev_in;
    uint8_t *dev_out;

    HANDLE_ERROR(cudaMalloc( &dev_in, MapWidth * MapWidth * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc( &dev_out, outMapWidth * outMapWidth * sizeof(uint8_t)));

    HANDLE_ERROR(cudaMemcpy(dev_in, h_in, MapWidth * MapWidth * sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_in_global, &dev_in, sizeof(uint8_t *)));
    


    dim3 gridDim(outMapWidth, outMapHeight);
    dim3 blockDim(ThrPerBlock_x, ThrPerBlock_y);
    ChronoGPU chrk1;
    chrk1.start();
    for (int i = 0; i < NbIteration; i++) kernelNaive_tiledGPU<<<gridDim, blockDim>>>(dev_out, MapWidth, MapHeight, outMapWidth, outMapHeight);
    chrk1.stop();

	const float timeComputechrk1 = chrk1.elapsedTime();
    printf("Done kernelTiled : %f ms\n", timeComputechrk1 / NbIteration);


    HANDLE_ERROR(cudaMemcpy(h_out, dev_out, outMapWidth * outMapWidth * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_in));
    HANDLE_ERROR(cudaFree(dev_out));
}