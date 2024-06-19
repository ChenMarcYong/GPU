#ifndef __CARTE_GPU__
#define __CARTE_GPU__


#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"
#include "los/ppm.hpp"

using namespace los;

__global__ void kernelMapV1(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy);
__global__ void kernelMapV2(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy);
__global__ void kernelMapOptimize(uint8_t *h_in, uint8_t *h_out, float *angle, const int MapWidth, const int MapHeight, const int Cx, const int Cy);


__global__ void kernelAngle(uint8_t *h_in, uint8_t *h_out, float *angle, const int MapWidth, const int MapHeight, const int Cx, const int Cy);


void carteGPUv1(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy);
void carteGPUv2(uint8_t *h_in, uint8_t *h_out, float *angle, const int MapWidth, const int MapHeight, const int Cx, const int Cy);

__device__ float angle(int Dx, int Dy, int Dz);


#endif //__CARTE_GPU__