#include "references/referenceGPU.hpp"

#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"

#include "main.hpp"


#define block_size 256
#define num_blocks 32

__global__ void histgramGPU_kernel(int *dataset, int *result, int elements_count){
	__shared__ unsigned int cache[DISTRIBUTION_SIZE];

    for(int x = blockDim.x * blockIdx.x + threadIdx.x; x < elements_count; x += blockDim.x * gridDim.x){
        int value = dataset[x];

		//cache[cacheIndex][value]++;
        atomicAdd(&cache[value], 1);
    }

	__syncthreads();

	if(threadIdx.x != 0){
		return;
	}

	for(int i = 0; i < DISTRIBUTION_SIZE; i++){
        atomicAdd(&result[i], cache[i]);
	}
}

float histogramGPU(int *dataset, int *result, int elements_count, int distribution_size){
	ChronoGPU chr;

	int *dev_dataset;
	int *dev_result;
    
	cudaMalloc((void **) &dev_dataset, elements_count * sizeof(int));
	cudaMalloc((void **) &dev_result, distribution_size * sizeof(int));

	cudaMemcpy(dev_dataset, dataset, elements_count * sizeof(int), cudaMemcpyHostToDevice);

	chr.start();
    histgramGPU_kernel<<<block_size, num_blocks>>>(dev_dataset, dev_result, elements_count);
	chr.stop();


	cudaMemcpy(result, dev_result, distribution_size * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_dataset);
	cudaFree(dev_result);

	return chr.elapsedTime();
}
