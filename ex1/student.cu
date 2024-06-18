#include "student.hpp"


#define DIM_BLOCK 32

__global__ void kernelHistograme( int *dev_tab, int size ,int* dev_res ) {
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += ( blockDim.x * gridDim.x )){
		int elem = dev_tab[i];
		atomicAdd(&dev_res[elem],1);
	}

}

float histogrameGPU( int *tab, int dist, int size, int* res )
{
	
	int* dev_tab, *dev_res;
	
	/// Allocate memory on Device
	HANDLE_ERROR( cudaMalloc( &dev_tab, sizeof(int) * size ) );
	HANDLE_ERROR( cudaMalloc( &dev_res, sizeof(int) * dist ) );


	/// Copy from Host to Device
	HANDLE_ERROR( cudaMemcpy( dev_tab, tab, sizeof(int) * size, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_res, res, sizeof(int) * dist, cudaMemcpyHostToDevice ) );

	/// Configure kernel
	dim3 dimBlock ( DIM_BLOCK );
	dim3 dimGrid  ( ( size + DIM_BLOCK - 1 ) / DIM_BLOCK );

	// Chrono
	ChronoGPU chr;
	chr.start();

	kernelHistograme <<< dimGrid, dimBlock >>> ( dev_tab, size, dev_res );

	// Chrono stop
	chr.stop();

	// Copy from Device to Host
	HANDLE_ERROR( cudaMemcpy( res, (void*)dev_res, sizeof(int) * dist, cudaMemcpyDeviceToHost ) );

	/// Free memory on Device
	HANDLE_ERROR( cudaFree( dev_res ) );
    HANDLE_ERROR( cudaFree( dev_tab ) );	

	return chr.elapsedTime();
}
