#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "optimized_viewsetGPU.hpp"



using namespace std;
#define ThrPerBlock_y 8
#define ThrPerBlock_x 8

__device__ __constant__ uint8_t* dev_in_global;
__device__ __constant__ uint8_t* angle_global;
texture<uint8_t, cudaTextureType2D, cudaReadModeElementType> texIn; // améliore le temps d'éxécution même si on utilise juste cette ligne de code
__device__ float calculateAngleOptimizedGPU(float Dz, float Dx, float Dy)
{
    float dist = sqrt( (Dx) * (Dx) + (Dy) * (Dy));
    return atan(Dz / dist);
}

__device__ uint8_t DDAOptimizedGPU(int Px, int Py, const int Cx, const int Cy, const int MapWidth) 
{
{
    int Dx, Dy, D;  // delta
    Dx = Cx - Px;
    Dy = Cy - Py;

    int Dz = dev_in_global[Cy * MapWidth + Cx] - dev_in_global[Py * MapWidth + Px];
    D = max(abs(Dx), abs(Dy));
    float angleRef;
    angleRef = calculateAngleOptimizedGPU(Dz, Dx, Dy);
    //angleRef = atan(Dz / __fsqrt_rn( (Dx * Dx) + (Dy * Dy)  )); 
    float angleDDA;

    float stepX, stepY;
    stepX = (float(Dx) / D);
    stepY = (float(Dy) / D);

    int DDAx, DDAy;
    for(int i = 0; i < D; i++)
    {
        DDAx = Px + i * stepX; 
        DDAy = Py + i * stepY;
        Dx = Cx - DDAx;
        Dy = Cy - DDAy;
        Dz = dev_in_global[Cy * MapWidth + Cx] - dev_in_global[DDAy * MapWidth + DDAx];
        //angleDDA = calculateAngleOptimizedGPU(Dz, Dx, Dy);
        angleDDA = atan(Dz / __fsqrt_rn( (Dx * Dx) + (Dy * Dy)  )); 

        if (angleRef > angleDDA) return 0;
    }
    return 255;
}
}




__global__ void kernelOptimized_viewsetGPU(uint8_t *dev_out, int Cx, int Cy, const int MapHeight, const int MapWidth)
{

    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int initX = indexX;

    while (indexY < MapHeight)
    {
        while (indexX < MapWidth)
        {
            dev_out[indexY * MapWidth + indexX] = DDAOptimizedGPU(indexX, indexY, Cx, Cy, MapWidth);
            indexX += gridDim.x * blockDim.x;
        }

        indexY += gridDim.y * blockDim.y;
        indexX = initX;
    }
}

__global__ void kernelAngleGPU(float *dev_angle,int Cx, int Cy, const int MapHeight, const int MapWidth)
{
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int initX = indexX;

    while (indexY < MapHeight)
    {
        while (indexX < MapWidth)
        {

            int Dx, Dy, D;  // delta
            Dx = Cx - indexX;
            Dy = Cy - indexY;

            int Dz = dev_in_global[Cy * MapWidth + Cx] - dev_in_global[indexY * MapWidth + indexX];
            D = max(abs(Dx), abs(Dy));
            float angleRef;
            dev_angle[indexY * MapWidth + indexX] = calculateAngleOptimizedGPU(Dz, Dx, Dy);
            indexX += gridDim.x * blockDim.x;
        }

        indexY += gridDim.y * blockDim.y;
        indexX = initX;
    }    
}



void optimized_viewsetGPU(const uint8_t *h_in, uint8_t *h_out, int Cx, int Cy, const int MapHeight, const int MapWidth) {
    uint8_t *dev_in, *dev_out;
    float *dev_angle;

    HANDLE_ERROR(cudaMalloc(&dev_in, sizeof(uint8_t) * MapHeight * MapWidth));
    HANDLE_ERROR(cudaMalloc(&dev_out, sizeof(uint8_t) * MapHeight * MapWidth));
    HANDLE_ERROR(cudaMalloc(&dev_angle, sizeof(float) * MapHeight * MapWidth));

    HANDLE_ERROR(cudaMemcpy(dev_in, h_in, sizeof(uint8_t) * MapHeight * MapWidth, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_out, h_out, sizeof(uint8_t) * MapHeight * MapWidth, cudaMemcpyHostToDevice));


    HANDLE_ERROR(cudaMemcpyToSymbol(dev_in_global, &dev_in, sizeof(uint8_t*)));

    int blocks_x = (MapWidth + ThrPerBlock_x - 1) / ThrPerBlock_x;
    int blocks_y = (MapHeight + ThrPerBlock_y - 1) / ThrPerBlock_y;

    dim3 gridDim(blocks_x, blocks_y);
    dim3 blockDim(ThrPerBlock_x, ThrPerBlock_y);

    kernelAngleGPU<<<gridDim, blockDim>>>(dev_angle, Cx, Cy, MapHeight, MapWidth);

    cudaDeviceSynchronize();

    HANDLE_ERROR(cudaMemcpyToSymbol(angle_global, &dev_angle, sizeof(float*)));

    HANDLE_ERROR(cudaFree(dev_angle));


    kernelOptimized_viewsetGPU<<<gridDim, blockDim>>>(dev_out, Cx, Cy, MapHeight, MapWidth);

    HANDLE_ERROR(cudaMemcpy(h_out, dev_out, sizeof(uint8_t) * MapHeight * MapWidth, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_in));
    HANDLE_ERROR(cudaFree(dev_out));

}



