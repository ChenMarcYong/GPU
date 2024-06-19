#ifndef __NAIVE_VIEWSET_GPU__
#define __NAIVE_VIEWSET_GPU__

#include "../../los/ppm.hpp"
#include "../../utils/commonCUDA.hpp"

using namespace los;


__device__ uint8_t NaiveGPU(const uint8_t *h_in, int Px, int Py, const int Cx, const int Cy, const int MapWidth);
__device__ float calculateAngleNaiveGPU(float Dz, float Dx, float Dy);

__global__ void kernelNaive_viewsetGPU(const uint8_t *h_in, uint8_t *h_out, int Cx, int Cy, const int MapHeight, const int MapWidth);
void naive_viewsetGPU(const uint8_t *h_in, uint8_t *h_out, int Cx, int Cy, const int MapHeight, const int MapWidth);

#endif //__NAIVE_VIEWSET_GPU__