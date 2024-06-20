#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "naive_viewsetCPU.hpp"

using namespace std;

float calculateAngle(float Dz, float Dx, float Dy)
{
    float dist = sqrt((Dx) * (Dx) + (Dy) * (Dy));
    return atan(Dz / dist);
}

uint8_t DDA(const uint8_t * h_in, int Px, int Py, const int Cx, const int Cy, const int MapWidth)
{
    int Dx, Dy, D;  // delta
    Dx = Px - Cx;
    Dy = Py - Cy;

    int Dz = h_in[Cy * MapWidth + Cx] - h_in[Py * MapWidth + Px];
    D = max(abs(Dx), abs(Dy));
    float angleRef;
    angleRef = calculateAngle(Dz, Dx, Dy);
    //angleRef = atan(Dz / sqrt( (Dx * Dx) + (Dy * Dy)  )); 
    float angleDDA;

    float stepX, stepY;
    stepX = (float(Dx) / D);
    stepY = (float(Dy) / D);

    int DDAx, DDAy;
    for(int i = 0; i < D; i++)
    {
        DDAx = Cx + i * stepX; 
        DDAy = Cy + i * stepY;
        Dx = Cx - DDAx;
        Dy = Cy - DDAy;
        Dz = h_in[Cy * MapWidth + Cx] - h_in[DDAy * MapWidth + DDAx];
        //angleDDA = calculateAngle(Dz, Dx, Dy);
        angleDDA = atan(Dz / sqrt( (Dx * Dx) + (Dy * Dy)  )); 

        if (angleRef > angleDDA) return 0;
    }
    return 255;
}

void naive_viewsetCPU(const uint8_t * h_in, uint8_t * h_out, int Cx, int Cy, const int MapHeight, const int MapWidth)
{
    for (int Py = 0; Py < MapHeight; Py++)
    {
        for (int Px = 0; Px < MapWidth; Px++)
        {
            h_out[Py * MapWidth + Px] = DDA(h_in, Px, Py, Cx, Cy, MapWidth);
        }
    }
}


