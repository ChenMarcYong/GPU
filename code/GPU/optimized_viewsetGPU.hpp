#ifndef __OPTIMIZED_VIEWSET_GPU__
#define __OPTIMIZED_VIEWSET_GPU__

#include "../../los/ppm.hpp"
#include "../../utils/commonCUDA.hpp"

using namespace los;


__device__ uint8_t DDAOptimizedGPU(int Px, int Py, const int Cx, const int Cy, const int MapWidth);
__device__ float calculateAngleOptimizedGPU(float Dz, float Dx, float Dy);

__global__ void kernelOptimized_viewsetGPU(uint8_t *h_out, int Cx, int Cy, const int MapHeight, const int MapWidth);
__global__ void kernelAngleGPU(float *dev_angle,int Cx, int Cy, const int MapHeight, const int MapWidth);
void optimized_viewsetGPU(const uint8_t *h_in, uint8_t *h_out, int Cx, int Cy, const int MapHeight, const int MapWidth);

#endif //__OPTIMIZED_VIEWSET_GPU__