#ifndef __NAIVE_VIEWSET_CPU__
#define __NAIVE_VIEWSET_CPU__

#include "../../los/ppm.hpp"

using namespace los;


uint8_t DDA(const uint8_t * h_in, int Px, int Py, const int Cx, const int Cy, const int MapWidth);
float calculateAngle(float Dz, float Dx, float Dy);

void naive_viewsetCPU(const uint8_t * h_in, uint8_t * h_out, int Cx, int Cy, const int MapHeight, const int MapWidth);

#endif //__NAIVE_VIEWSET_CPU__