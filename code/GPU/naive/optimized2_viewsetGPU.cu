#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "optimized2_viewsetGPU.hpp"

#include "../../../utils/chronoGPU.hpp"



using namespace std;
#define ThrPerBlock_y 16
#define ThrPerBlock_x 16
#define NbIteration 1000


__device__ __constant__ uint8_t* dev_in_global;
__device__ __constant__ float* angle_global;
__device__ float calculateAngleOptimized2GPU(float Dz, float Dx, float Dy)
{
    float dist = sqrt( (Dx) * (Dx) + (Dy) * (Dy));
    return atan(Dz / dist);
}

__device__ void DDAOptimized2GPU(uint8_t *dev_out, int Px, int Py, const int Cx, const int Cy, const int MapWidth) 
{
    int Dx, Dy, D;  // delta
    Dx = Px - Cx;
    Dy = Py - Cy;

    D = max(abs(Dx), abs(Dy));
    float angleMax = - M_PI/2;
    float angleDDA;

    float stepX, stepY;
    stepX = (float(Dx) / D);
    stepY = (float(Dy) / D);

    int DDAx, DDAy;
    for(int i = 0; i <= D; i++)
    {
        DDAx = Cx + i * stepX; 
        DDAy = Cy + i * stepY;
        angleDDA = angle_global[DDAy * MapWidth + DDAx]; 
        if (angleMax < angleDDA)
        {
            dev_out[DDAy * MapWidth + DDAx] = 255;
            angleMax = angleDDA;
        } 
    }
}




__global__ void kernelOptimized2_viewsetGPU(uint8_t *dev_out, int Cx, int Cy, const int MapHeight, const int MapWidth)
{

    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int initX = indexX;

    while (indexY <= MapHeight)
    {
        while (indexX <= MapWidth)
        {
            if (indexX == 0 || indexY == 0 || indexX == MapWidth || indexY == MapHeight)
            {
                DDAOptimized2GPU(dev_out, indexX, indexY, Cx, Cy, MapWidth);
            }
            
            indexX += gridDim.x * blockDim.x;
        }

        indexY += gridDim.y * blockDim.y;
        indexX = initX;
    }
}

__global__ void kernelAngle2GPU(float *dev_angle, int Cx, int Cy, const int MapHeight, const int MapWidth)
{
    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;
    int initX = indexX;

    while (indexY < MapHeight)
    {
        while (indexX < MapWidth)
        {
            
            int Dx, Dy;  // delta
            Dx = indexX - Cx;
            Dy = indexY - Cy;

            int Dz = dev_in_global[indexY * MapWidth + indexX] - dev_in_global[Cy * MapWidth + Cx];
            dev_angle[indexY * MapWidth + indexX] = calculateAngleOptimized2GPU(Dz, Dx, Dy);
            indexX += gridDim.x * blockDim.x;
        }

        indexY += gridDim.y * blockDim.y;
        indexX = initX;
    }    
}



void optimized2_viewsetGPU(const uint8_t *h_in, uint8_t *h_out, int Cx, int Cy, const int MapHeight, const int MapWidth) {
    uint8_t *dev_in, *dev_out;
    float *dev_angle;

    HANDLE_ERROR(cudaMalloc(&dev_angle, sizeof(float) * MapHeight * MapWidth));
    HANDLE_ERROR(cudaMalloc(&dev_in, sizeof(uint8_t) * MapHeight * MapWidth));
    HANDLE_ERROR(cudaMalloc(&dev_out, sizeof(uint8_t) * MapHeight * MapWidth));
    
    HANDLE_ERROR(cudaMemcpy(dev_in, h_in, sizeof(uint8_t) * MapHeight * MapWidth, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_in_global, &dev_in, sizeof(uint8_t *)));
    

    int blocks_x = (MapWidth + ThrPerBlock_x - 1) / ThrPerBlock_x;
    int blocks_y = (MapHeight + ThrPerBlock_y - 1) / ThrPerBlock_y;

    dim3 gridDim(blocks_x, blocks_y);
    dim3 blockDim(ThrPerBlock_x, ThrPerBlock_y);

    ChronoGPU chrk1;
    chrk1.start();
	for (int i = 0; i < NbIteration; i++)
	{
        kernelAngle2GPU<<<gridDim, blockDim>>>(dev_angle, Cx, Cy, MapHeight, MapWidth);

        cudaDeviceSynchronize();

        HANDLE_ERROR(cudaMemcpyToSymbol(angle_global, &dev_angle, sizeof(float *)));

        kernelOptimized2_viewsetGPU<<<gridDim, blockDim>>>(dev_out, Cx, Cy, MapHeight, MapWidth);
    }
    chrk1.stop();
    const float timeComputechrk1 = chrk1.elapsedTime();
    printf("Done optimized2_KernelviewsetGPU : %f ms\n", timeComputechrk1 / NbIteration);


    HANDLE_ERROR(cudaMemcpy(h_out, dev_out, sizeof(uint8_t) * MapHeight * MapWidth, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_in));
    HANDLE_ERROR(cudaFree(dev_out));
    HANDLE_ERROR(cudaFree(dev_angle));

}



