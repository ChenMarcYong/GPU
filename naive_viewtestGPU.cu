#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"

#include "naive_viewtestGPU.hpp"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"

using namespace std;
#define ThrPerBlock_y 8
#define ThrPerBlock_x 8

__global__ void kernelMapV1(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    for(int Py = blockDim.y * blockIdx.y + threadIdx.y; Py < MapHeight; Py += blockDim.y * gridDim.y)
    {
        for(int Px = blockDim.x * blockIdx.x + threadIdx.x; Px < MapWidth; Px += blockDim.x * gridDim.x)
        {
            int Dx, Dy, Dz,  D;
            Dx = Px - Cx;   // delta x
            Dy = Py - Cy;   // delta y
            Dz = h_in[Py * MapWidth + Px] - h_in[Cy * MapWidth + Cx];   // delta z
            D = max(abs(Dx), abs(Dy));  // delta positif max entre Dx et Dy
            float incX = ((float)Dx / D), incY = ((float)Dy / D);
            float angle_ref = atan(Dz / __fsqrt_rn((Dx * Dx) + (Dy * Dy)));
            h_out[Py * MapWidth + Px] = 255;
            for(int i = 0; i < D; i++)      // boucle dans la dda
            {
                int x = Cx + i * incX;
                int y = Cy + i * incY;
                Dx = Px - x; 
                Dy = Py - y;
                Dz = h_in[Py * MapWidth + Px] - h_in[y * MapWidth + x]; 
                float angle = atan(Dz / __fsqrt_rn((Dx * Dx) + (Dy * Dy)));
                if (angle_ref > angle)
                {
                    h_out[Py * MapWidth + Px] = 0;
                    break;
                }
            }
        }
    }
}

__global__ void kernelMapV2(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    __shared__ uint8_t tile[ThrPerBlock_y][ThrPerBlock_x];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;

    tile[ty][tx] = 255;

    __syncthreads();

    int Dx, Dy, Dz, D;
    Dx = x - Cx;
    Dy = y - Cy;
    Dz = h_in[y * MapWidth + x] - h_in[Cy * MapWidth + Cx];
    D = max(abs(Dx), abs(Dy));
    float incX = ((float)Dx / D), incY = ((float)Dy / D);
    float angle_ref = atan(Dz / __fsqrt_rn((Dx * Dx) + (Dy * Dy)));

    for (int i = 0; i < D; i++)
    {
        int x_dda = Cx + i * incX;
        int y_dda = Cy + i * incY;
        Dx = x - x_dda;
        Dy = y - y_dda;
        Dz = h_in[y * MapWidth + x] - h_in[y_dda * MapWidth + x_dda];
        float angle = atan(Dz / __fsqrt_rn((Dx * Dx) + (Dy * Dy)));

        if (angle_ref > angle)
        {
            tile[ty][tx]  = 0;
            break;
        }
    }
    __syncthreads();
    h_out[y * MapWidth + x] = tile[ty][tx];
}

__global__ void kernelAngle(uint8_t *h_in, uint8_t *h_out, float *angle, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{

    int Dx, Dy, Dz;
    for(int Py = blockDim.y * blockIdx.y + threadIdx.y; Py < MapHeight; Py += blockDim.y * gridDim.y)
    {
        for(int Px = blockDim.x * blockIdx.x + threadIdx.x; Px < MapWidth; Px += blockDim.x * gridDim.x)
        {
            Dx = Px - Cx;
            Dy = Py - Cy;
            Dz = h_in[Py * MapWidth + Px] - h_in[Cy * MapWidth + Cx];
            angle[Py * MapWidth + Px] = atan(Dz / __fsqrt_rn((Dx * Dx) + (Dy * Dy)));
        }
    }
}



__global__ void kernelMapOptimize(uint8_t *h_in, uint8_t *h_out, float *angle, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    int Dx, Dy, Dz, D;

    for(int Py = blockDim.y * blockIdx.y + threadIdx.y; Py <= MapHeight; Py += blockDim.y * gridDim.y)
    {
        for(int Px = blockDim.x * blockIdx.x + threadIdx.x; Px <= MapWidth; Px += blockDim.x * gridDim.x)
        {
            if (Px == 0 || Py == 0 || Px == MapWidth || Py == MapHeight)
            {
                Dx = Px - Cx;
                Dy = Py - Cy;
                D = max(abs(Dx), abs(Dy));
                float incX = ((float)Dx / D), incY = ((float)Dy / D);
                float angle_max = - M_PI/2;         // min de arctan
                for (int i = 0; i < D; i++)
                {
                    int hx = Cx + i * incX;
                    int hy = Cy + i * incY;
                    float current_angle = angle[hy * MapWidth + hx];         //calcAngle(Dx, Dy, Dz);
                    if (angle_max < current_angle)  
                    {
                        h_out[hy * MapWidth + hx] = 255;
                        angle_max = current_angle;
                    }
                }
            }
        }
    }
}


void carteGPUv1(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    ChronoGPU chrGPU;
    uint8_t *dev_h_in;
    uint8_t *dev_h_out;

    size_t size = MapWidth * MapHeight * sizeof(uint8_t);

    cudaMalloc((void**) &dev_h_in, size);
    cudaMalloc((void**) &dev_h_out, size);

    cudaMemcpy(dev_h_in, h_in, size, cudaMemcpyHostToDevice);

    int blocks_x = (MapWidth + ThrPerBlock_x - 1) / ThrPerBlock_x;
    int blocks_y = (MapHeight + ThrPerBlock_y - 1) / ThrPerBlock_y;

    dim3 gridDim(blocks_x, blocks_y);
    dim3 blockDim(ThrPerBlock_x, ThrPerBlock_y);

    ChronoGPU chrk1;
    chrk1.start();
    kernelMapV2<<<gridDim, blockDim>>>(dev_h_in, dev_h_out, MapWidth, MapHeight, Cx, Cy);
    chrk1.stop();

	const float timeComputechrk1 = chrk1.elapsedTime();
    printf("Done kernelMap : %f ms\n", timeComputechrk1);


    cudaMemcpy(h_out, dev_h_out, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_h_in);
    cudaFree(dev_h_out);
}










void carteGPUv2(uint8_t *h_in, uint8_t *h_out, float *angle, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    ChronoGPU chrGPU;
    uint8_t *dev_h_in;
    uint8_t *dev_h_out;
    float *dev_angle;

    size_t size = MapWidth * MapHeight * sizeof(uint8_t);
    size_t size_angle = MapWidth * MapHeight * sizeof(float);

    cudaMalloc((void**) &dev_h_in, size);
    cudaMalloc((void**) &dev_h_out, size);
    cudaMalloc((void**) &dev_angle, size_angle);

    cudaMemcpy(dev_h_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_angle, angle, size_angle, cudaMemcpyHostToDevice);

    int blocks_x =(int) (((MapWidth + ThrPerBlock_x - 1) / ThrPerBlock_x));
    int blocks_y = (int)(((MapHeight + ThrPerBlock_y - 1) / ThrPerBlock_y));

    dim3 gridDim(blocks_x, blocks_y);
    dim3 blockDim(ThrPerBlock_x, ThrPerBlock_y);

	ChronoGPU chrk1;
	ChronoGPU chrk2;
	chrk1.start();		// CPU method
    kernelAngle<<<gridDim, blockDim>>>(dev_h_in, dev_h_out, dev_angle, MapWidth, MapHeight, Cx, Cy);
	chrk1.stop();
	
	const float timeComputechrk1 = chrk1.elapsedTime();
    printf("Done kernelAngle : %f ms\n", timeComputechrk1);
    chrk2.start();
    kernelMapOptimize<<<gridDim, blockDim>>>(dev_h_in, dev_h_out, dev_angle, MapWidth, MapHeight, Cx, Cy);
    chrk2.stop();
    const float timeComputechrk2 = chrk2.elapsedTime();
    printf("Done kernelMap : %f ms\n", timeComputechrk2);


    cudaMemcpy(h_out, dev_h_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(angle, dev_angle, size_angle, cudaMemcpyDeviceToHost);

    cudaFree(dev_h_in);
    cudaFree(dev_h_out);
    cudaFree(dev_angle);
}