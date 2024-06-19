#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "naive_viewtestCPU.hpp"
#include "los/ppm.hpp"

using namespace std;


void drawMap(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx,const int Cy)
{
    for(int Py = 0; Py < MapHeight; Py++)
    {
        for(int Px = 0; Px < MapWidth; Px++)
        {

        // DDA entre le point c (Cx, Cy) et le point P (Px, Py);
            float Dx, Dy, Dz,  D;

            Dx = Px - Cx;   // delta x
            Dy = Py - Cy;   // delta y
            Dz = h_in[Py * MapWidth + Px] - h_in[Cy * MapWidth + Cx];   // delta z
            D = max(abs(Dx), abs(Dy));  // delta positif max entre Dx et Dy
            float angle, angle_ref = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));

            float incX = (Dx / D);
            float incY = (Dy / D);
            h_out[Py * MapWidth + Px] = 255;
            for (int i = 0; i < D; i++)
            {
                int x = Cx + incX * i;
                int y = Cy + incY * i;
                Dx = Px - x;
                Dy = Py - y;    
                Dz = h_in[Py * MapWidth + Px] - h_in[y * MapWidth + x];  
                // Calcule Angle 
                angle = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));     
                if (angle_ref >= angle)
                {
                    //h_out.setPixel(Px, Py, 0);
                    h_out[Py * MapWidth + Px] = 0;
                    break;
                }                         
            }   
        }
    }
}

// regarder les points en bordure arctan min = -pi/2 et arctan max = pi/2
// boucler sur chaque points en bordure et effectuer une dda sur chaque points, comme

void drawMap2(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    for(int Py = 0; Py < MapHeight; Py++)
    {
        if (Py == 0 || Py == MapHeight - 1)
        {
            for(int Px = 0; Px < MapHeight; Px++)
            {
                float Dx, Dy, Dz,  D;
                Dx = Px - Cx;   // delta x
                Dy = Py - Cy;   // delta y
                Dz = h_in[Py * MapWidth + Px] - h_in[Cy * MapWidth + Cx];   // delta z
                D = max(abs(Dx), abs(Dy));  // delta positif max entre Dx et Dy
                double angle, angle_ref = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));

                float Cx_dda = (float) Cx, Cy_dda = (float) Cy;
                int Lx, Ly;
                float incX = (Dx / D);
                float incY = (Dy / D);
                h_out[Py * MapWidth + Px] = 244;
                for (int i = 0; i < D - 1; i++)
                {

                    Cx_dda += incX;
                    Cy_dda += incY;
                    Lx = (int)round(Cx_dda);
                    Ly = (int)round(Cy_dda);
                    Dx = Px - Lx;
                    Dy = Py - Ly;    
                    Dz = h_in[Py * MapWidth + Px] - h_in[Ly * MapWidth + Lx];  
                    // Calcule Angle 
                    angle = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));     
                    if (angle_ref >= angle)
                    {
                        //h_out.setPixel(Px, Py, 0);
                        h_out[Py * MapWidth + Px] = 0;
                        break;
                    }                         
                }   
            }
        }
    }
}

void drawMap3(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    for(int Py = 0; Py < MapHeight; Py++)
    {
        for(int Px = 0; Px < MapWidth ; Px++)
        {
            int Dx, Dy, Dz,  D;
            Dx = Px - Cx;   // delta x
            Dy = Py - Cy;   // delta y
            Dz = h_in[Py * MapWidth + Px] - h_in[Cy * MapWidth + Cx];   // delta z
            D = max(abs(Dx), abs(Dy));  // delta positif max entre Dx et Dy
            float incX = ((float)Dx / D), incY = ((float)Dy / D);
            float angle_ref = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));

            //if (Py == 0) std::cout << angle_ref << " " << std::endl;
            //std::cout << incX << "  " << incY << std::endl;
            h_out[Py * MapWidth + Px] = 255;
            for(int i = 0; i < D; i++)      // boucle dans la dda
            {
                int x = Cx + i * incX;
                int y = Cy + i * incY;
                Dx = Px - x; 
                Dy = Py - y;
                Dz = h_in[Py * MapWidth + Px] - h_in[y * MapWidth + x]; 
                float angle = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));
                if (angle_ref > angle)
                {
                    h_out[Py * MapWidth + Px] = 0;
                    break;
                }
                //std::cout << x << "  " << y << "      " << i << std::endl;
            }
        }
    }
}