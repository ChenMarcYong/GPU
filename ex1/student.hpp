#ifndef __STUDENT_HPP__
#define __STUDENT_HPP__

#include "utils/ppm.hpp"
#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"


__global__ void kernelHistograme( int *dev_tab, int size,int* dev_res );

float histogrameGPU( int *tab, int dist, int size, int* res );
#endif // __STUDENT_HPP__
