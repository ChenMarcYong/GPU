#include "references/referenceCPU.hpp"

#include "utils/chronoCPU.hpp"

void histgramCPU_kernel(int *dataset, int *result, int elements_count){
    for(int i = 0; i < elements_count; i++){
        int value = dataset[i];

        result[value]++;
    }   
}

float histogramCPU(int *dataset, int *result, int elements_count){
	ChronoCPU chr;

	chr.start();
    histgramCPU_kernel(dataset, result, elements_count);
	chr.stop();

	return chr.elapsedTime();
}