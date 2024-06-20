#ifndef __OPTIMIZED2_VIEWSET_GPU__
#define __OPTIMIZED2_VIEWSET_GPU__

#include "../../../los/ppm.hpp"
#include "../../../utils/commonCUDA.hpp"

using namespace los;


__device__ void DDAOptimized2GPU(uint8_t *dev_out, int Px, int Py, const int Cx, const int Cy, const int MapWidth);
__device__ float calculateAngleOptimized2GPU(float Dz, float Dx, float Dy);

__global__ void kernelOptimized2_viewsetGPU(uint8_t *h_out, int Cx, int Cy, const int MapHeight, const int MapWidth);
__global__ void kernelAngle2GPU(float *dev_angle,int Cx, int Cy, const int MapHeight, const int MapWidth);
void optimized2_viewsetGPU(const uint8_t *h_in, uint8_t *h_out, int Cx, int Cy, const int MapHeight, const int MapWidth);

#endif //__OPTIMIZED2_VIEWSET_GPU__