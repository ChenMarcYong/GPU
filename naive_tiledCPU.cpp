#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "naive_tiledCPU.hpp"
#include "los/ppm.hpp"


uint8_t maxHeight(uint8_t *h_in, const int DimX, const int DimY)
{
    uint8_t maxHeight = 0;
    for(int y = 0; y < DimY; y++)
    {

    }
    return maxHeight;

}


void tiledMap(uint8_t *h_in, uint8_t *h_out,const int inMapWidth, const int inMapHeight , const int outMapWidth, const int outMapHeight)
{
    int TiledDimX = int(inMapWidth / outMapWidth);
    int TiledDimY = int(inMapHeight / outMapHeight);

    for(int TiledIdY = 0; TiledIdY < outMapHeight; TiledIdY++)
    {
        for(int TiledIdX = 0; TiledIdX < outMapWidth; TiledIdX++)
        {
            uint8_t maxHeight = 0;
            int location = (TiledIdY * TiledDimY) * (inMapWidth) + TiledIdX * TiledDimX;
            for(int y = 0; y < TiledDimY; y++)
            {
                for(int x = 0; x < TiledDimX; x++)
                {
                    int point = location + y * inMapWidth + x; 
                    if (h_in[point] > maxHeight) maxHeight = h_in[point];
                }               
            }
            h_out[TiledIdY * outMapWidth + TiledIdX] = maxHeight;
        }
    }
}
