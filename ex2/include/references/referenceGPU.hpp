#ifndef REFERENCE_GPU_HPP
#define REFERENCE_GPU_HPP

__global__ void histgramGPU_kernel(int *dataset, int *result, int elements_count);
float histogramGPU(int *dataset, int *result, int elements_count, int distribution_size);

#endif // REFERENCE_GPU_HPP