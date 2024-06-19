#ifndef __TILED_GPU__
#define __TILED_GPU__


#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"
#include "los/ppm.hpp"

using namespace los;

__global__ void kernelTiled(uint8_t *h_in, uint8_t *h_out,const int inMapWidth, const int inMapHeight , const int outMapWidth, const int outMapHeight);

void TiledGPU(uint8_t *h_in, uint8_t *h_out,const int inMapWidth, const int inMapHeight , const int outMapWidth, const int outMapHeight);


#endif //__TILED_GPU__