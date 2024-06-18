#include <iostream>
#include <string>
#include <iomanip>

#include "utils/ppm.hpp"
#include "reference.hpp"
#include "student.hpp"
#include "utils/chronoCPU.hpp"
#include "generator.hpp"
#include <vector>

#define SIZE 100*1000*1000
#define N 128


int main( int argc, char **argv ) 
{	
	
	// ===============================================================================================
	// Generation
	std::vector<int> tab(SIZE);
	hst::Generator g;

	g.sample(SIZE, N, tab.data());


	// ================================================================================================================
	// CPU sequential
	std::cout << "============================================"	<< std::endl;
	std::cout << "         Sequential version on CPU          "	<< std::endl;
	std::cout << "============================================"	<< std::endl;

	ChronoCPU chr;
	
	std::vector<int> hist(N, 0);

	chr.start();

	histogrameCPU(tab.data(), SIZE, hist.data());

	chr.stop();
	for(int i=0; i<N; i++)
		std::cout << hist[i] << "   ";
	std::cout << std::endl;

	const float timeCPU = chr.elapsedTime();
	
	std::cout << "-> Done : " << timeCPU << " ms" << std::endl << std::endl;


	// ================================================================================================================
	// GPU CUDA
	std::cout << "============================================"	<< std::endl;
	std::cout << "         Parallel version on GPU            "	<< std::endl;
	std::cout << "============================================"	<< std::endl;

	std::vector<int> hist2(N, 0);

	const float timeGPU = histogrameGPU(tab.data(), N, SIZE, hist2.data());

	for(int i=0; i<N; i++) 
		std::cout << hist2[i] << "   ";
	std::cout << std::endl;
	
	std::cout << "-> Done : " << timeGPU << " ms" << std::endl << std::endl;
	
	// ================================================================================================================

	std::cout << "Congratulations! Job's done!" << std::endl << std::endl;

	std::cout << "============================================" << std::endl;
	std::cout << "           Times recapitulation             " << std::endl;
	std::cout << "============================================" << std::endl;
	std::cout << "-> CPU: " << std::fixed << std::setprecision(2) << timeCPU << " ms" << std::endl;
	std::cout << "-> GPU: " << std::fixed << std::setprecision(2) << timeGPU << " ms" << std::endl;

	return EXIT_SUCCESS;
}