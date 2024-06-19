#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "naive_viewsetGPU.hpp"



using namespace std;
#define ThrPerBlock_y 8
#define ThrPerBlock_x 8

__device__ float calculateAngleNaiveGPU(float Dz, float Dx, float Dy)
{
    float dist = sqrt( (Dx) * (Dx) + (Dy) * (Dy));
    return atan(Dz / dist);
}

__device__ uint8_t DDANaiveGPU(const uint8_t *dev_in, int Px, int Py, const int Cx, const int Cy, const int MapWidth)
{
    int Dx, Dy, D;  // delta
    Dx = Px - Cx;
    Dy = Py - Cy;

    int Dz = dev_in[Py * MapWidth + Px] - dev_in[Cy * MapWidth + Cx];
    D = max(abs(Dx), abs(Dy));
    float angleRef;
    angleRef = calculateAngleNaiveGPU(Dz, Dx, Dy);
    //angleRef = atan(Dz / __fsqrt_rn( (Dx * Dx) + (Dy * Dy)  )); 
    float angleDDA;

    float stepX, stepY;
    stepX = (float(Dx) / D);
    stepY = (float(Dy) / D);

    int DDAx, DDAy;
    for(int i = 0; i < D; i++)
    {
        DDAx = Cx + i * stepX; 
        DDAy = Cy + i * stepY;
        Dx = Px - DDAx;
        Dy = Py - DDAy;
        Dz = dev_in[Py * MapWidth + Px] - dev_in[DDAy * MapWidth + DDAx];
        //angleDDA = calculateAngleNaiveGPU(Dz, Dx, Dy);
        angleDDA = atan(Dz / __fsqrt_rn( (Dx * Dx) + (Dy * Dy)  )); 

        if (angleRef > angleDDA) return 0;
    }
    return 255;
}

__global__ void kernelNaive_viewsetGPU(const uint8_t *dev_in, uint8_t *dev_out, int Cx, int Cy, const int MapHeight, const int MapWidth)
{

    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int initX = indexX;

    while (indexY < MapHeight)
    {
        while (indexX < MapWidth)
        {
            dev_out[indexY * MapWidth + indexX] = DDANaiveGPU(dev_in, indexX, indexY, Cx, Cy, MapWidth);
            indexX += gridDim.x * blockDim.x;
        }

        indexY += gridDim.y * blockDim.y;
        indexX = initX;
    }
}

void naive_viewsetGPU(const uint8_t *h_in, uint8_t *h_out, int Cx, int Cy, const int MapHeight, const int MapWidth)
{
    uint8_t *dev_in, *dev_out;

    HANDLE_ERROR(cudaMalloc(&dev_in, sizeof(uint8_t) * MapHeight * MapWidth));
    HANDLE_ERROR(cudaMalloc(&dev_out, sizeof(uint8_t) * MapHeight * MapWidth));

    HANDLE_ERROR(cudaMemcpy(dev_in, h_in, sizeof(uint8_t) * MapHeight * MapWidth, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_out, h_out, sizeof(uint8_t) * MapHeight * MapWidth, cudaMemcpyHostToDevice));


    int blocks_x = (MapWidth + ThrPerBlock_x - 1) / ThrPerBlock_x;
    int blocks_y = (MapHeight + ThrPerBlock_y - 1) / ThrPerBlock_y;

    dim3 gridDim(blocks_x, blocks_y);
    dim3 blockDim(ThrPerBlock_x, ThrPerBlock_y);

    kernelNaive_viewsetGPU<<<gridDim, blockDim >>>(dev_in, dev_out, Cx, Cy, MapHeight, MapWidth);

    HANDLE_ERROR(cudaMemcpy(h_out, dev_out, sizeof(uint8_t) * MapHeight * MapWidth, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_in));
    HANDLE_ERROR(cudaFree(dev_out));
}


