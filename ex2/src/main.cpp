#include "main.hpp"
#include "generator.hpp"

#include <iostream>
#include <cstdlib>
#include <iomanip>

#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"

#include "references/referenceCPU.hpp"
#include "references/referenceGPU.hpp"

void print_first_5_elements(int *values){
    for(int i = 0; i < 5; i++){
        std::cout << values[i] << std::endl;
    }
}

bool compareTwoArrays(int *a, int *b){
    for(int i = 0; i < DATASET_SIZE; i++){
        if(a[i] != b[i]){
            return false;
        }
    }

    return true;
}

int main(int /* argc */, char ** /* argv */)
{
	// ================================================================================================================
    // Initialization
    std::cout << "Allocating " << ((DATASET_SIZE * sizeof(int)) >> 20) << " MB on Host" << std::endl;

    int *dataset = new int[DATASET_SIZE];
    int* resultCPU = new int[DATASET_SIZE];
    int* resultGPU = new int[DATASET_SIZE];

    hst::Generator generator;
    generator.sample(DATASET_SIZE, DISTRIBUTION_SIZE, dataset);
    
	// ================================================================================================================
	// CPU sequential
	std::cout << "============================================" << std::endl;
	std::cout << "         Sequential version on CPU          " << std::endl;
	std::cout << "============================================" << std::endl;

    const float timeComputeCPU = histogramCPU(dataset, resultCPU, DATASET_SIZE);

	std::cout << "-> Done : " << std::fixed << std::setprecision(2) << timeComputeCPU << " ms" << std::endl
			  << std::endl;

	// ================================================================================================================
	// GPU CUDA
	std::cout << "============================================" << std::endl;
	std::cout << "          Parallel version on GPU           " << std::endl;
	std::cout << "============================================" << std::endl;

    const float timeComputeGPU = histogramGPU(dataset, resultGPU, DATASET_SIZE, DISTRIBUTION_SIZE);
    
	std::cout << "-> Done : " << timeComputeGPU << " ms" << std::endl
			  << std::endl;

    // ================================================================================================================

	std::cout << "============================================" << std::endl;
	std::cout << "              Checking results              " << std::endl;
	std::cout << "============================================" << std::endl;

	bool isEqual = compareTwoArrays(resultCPU, resultGPU);
    if(!isEqual){
        std::cerr << "Retry!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Congratulations! Job's done!" << std::endl
			  << std::endl;

	std::cout << "============================================" << std::endl;
	std::cout << "     Times recapitulation (only hist)       " << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "-> CPU: " << std::fixed << std::setprecision(2) << timeComputeCPU << " ms" << std::endl;
	std::cout << "-> GPU: " << std::fixed << std::setprecision(2) << timeComputeGPU << " ms" << std::endl;

	return EXIT_SUCCESS;
}
