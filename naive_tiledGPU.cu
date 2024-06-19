#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"

#include "naive_tiledGPU.hpp"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"

using namespace std;
#define ThrPerBlock_y 8
#define ThrPerBlock_x 8

__global__ void kernelTiled(uint8_t *h_in, uint8_t *h_out,const int inMapWidth, const int inMapHeight , const int outMapWidth, const int outMapHeight)
{
    int TiledDimX = int(inMapWidth / outMapWidth);
    int TiledDimY = int(inMapHeight / outMapHeight);

    __shared__ int max;

    int location = (blockIdx.y * TiledDimY) * (inMapWidth) + blockIdx.x * TiledDimX;

    for(int Py =threadIdx.y; Py < TiledDimY; Py += blockDim.y)
    {
        for(int Px =threadIdx.x; Px < TiledDimX; Px += blockDim.x)
        {
            int point = location + Py * inMapWidth + Px; 
            atomicMax(&max, int(h_in[point]));
        }               
    }
    __syncthreads();
    h_out[blockIdx.y * outMapWidth + blockIdx.x] = max;
}


void TiledGPU(uint8_t *h_in, uint8_t *h_out,const int inMapWidth, const int inMapHeight , const int outMapWidth, const int outMapHeight)
{
    ChronoGPU chrGPU;
    uint8_t *dev_h_in;
    uint8_t *dev_h_out;

    size_t sizeIn = inMapWidth * inMapWidth * sizeof(uint8_t);
    size_t sizeOut = outMapWidth * outMapWidth * sizeof(uint8_t);

    cudaMalloc((void**) &dev_h_in, sizeIn);
    cudaMalloc((void**) &dev_h_out, sizeOut);

    cudaMemcpy(dev_h_in, h_in, sizeIn, cudaMemcpyHostToDevice);
    

    int blocks_x = (inMapWidth + ThrPerBlock_x - 1) / ThrPerBlock_x;
    int blocks_y = (inMapHeight + ThrPerBlock_y - 1) / ThrPerBlock_y;

    dim3 gridDim(10, 10);
    dim3 blockDim(ThrPerBlock_x, ThrPerBlock_y);
    ChronoGPU chrk1;
    chrk1.start();
    kernelTiled<<<gridDim, blockDim>>>(dev_h_in, dev_h_out, inMapWidth, inMapHeight, outMapWidth, outMapHeight);
    chrk1.stop();

	const float timeComputechrk1 = chrk1.elapsedTime();
	//std::cout << "-> Done kernelAngle : " << chrk1 << " ms" << std::endl;
    printf("Done kernelTiled : %f ms\n", timeComputechrk1);


    cudaMemcpy(h_out, dev_h_out, sizeOut, cudaMemcpyDeviceToHost);

    cudaFree(dev_h_in);
    cudaFree(dev_h_out);
}