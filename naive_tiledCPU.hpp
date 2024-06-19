#ifndef __TILED_CPU__
#define __TILED_CPU__

#include "los/ppm.hpp"

using namespace los;

void tiledMap(uint8_t *h_in, uint8_t *h_out,const int inMapWidth, const int inMapHeight , const int outMapWidth, const int outMapHeight);
uint8_t maxHeight(uint8_t *h_in, const int DimX, const int DimY);


#endif //__TILED_CPU__