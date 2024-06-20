#ifndef __NAIVE_TILED_GPU__
#define __NAIVE_TILED_GPU__

#include "../../../los/ppm.hpp"
#include "../../../utils/commonCUDA.hpp"

using namespace los;

__global__ void kernelNaive_tiledGPU(uint8_t *h_out,const int MapWidth, const int MapHeight , const int outMapWidth, const int outMapHeight);
void naive_tiledGPU(uint8_t *h_in, uint8_t *h_out,const int MapWidth, const int MapHeight , const int outMapWidth, const int outMapHeight);

#endif //__NAIVE_TILED_GPU__