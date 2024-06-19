#ifndef __CARTE_CPU__
#define __CARTE_CPU__

#include "los/ppm.hpp"

using namespace los;

void drawMap(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy);
void drawMap2(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy);
void drawMap3(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy);
double calculateAngle(int Cx, int Cy, int Px, int Py, const uint8_t * h_in, const int MapWidth);
#endif //__CARTE_CPU__