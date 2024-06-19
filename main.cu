#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <vector>

#include "utils/chronoCPU.hpp"
#include "utils/chronoGPU.hpp"

#include "los/ppm.hpp"

#include "naive_viewtestCPU.hpp"
#include "naive_tiledCPU.hpp"

#include "naive_viewtestGPU.hpp"
#include "naive_tiledGPU.hpp"

#define Cx 245				// (245, 497)
#define Cy 497

#define TiledWidth 10
#define TiledHeight 10

int main(int argc, char **argv)
{
	// Parse program arguments
	// ================================================================================================================
	// Allocation and initialization

	// ================================================================================================================

	// ================================================================================================================
	// CPU sequential
	std::cout << "============================================" << std::endl;
	std::cout << "         Sequential version on CPU          " << std::endl;
	std::cout << "============================================" << std::endl;



    Heightmap h_inCPU("img/input/1.input.ppm");
    Heightmap h_outCPU(h_inCPU.getWidth(), h_inCPU.getHeight());

	ChronoGPU chrCPU;
	chrCPU.start();		// CPU method
	drawMap3(h_inCPU.getPtr(), h_outCPU.getPtr(), h_inCPU.getWidth(), h_inCPU.getHeight(),  Cx, Cy);
	//void drawMap(int *data,uint8_t *h_data, Heightmap h_out, const int MapSize, const int MapWidth, const int MapHeight,const int Cx,const int Cy)
	chrCPU.stop();
	h_outCPU.saveTo("img/Result/CPU/LimousinCPU.ppm");
	

	const float timeComputeCPU = chrCPU.elapsedTime();
	std::cout << "-> Done : " << std::fixed << std::setprecision(2) << timeComputeCPU << " ms" << std::endl
			  << std::endl;

    Heightmap h_inTiledCPU("img/input/1.input.ppm");
	Heightmap tiled("img/input/3.tiled.ppm");
    Heightmap h_outTiledCPU(tiled.getWidth(), tiled.getHeight());
	//std::cout << tiled.getWidth() << " " << tiled.getHeight() << std::endl;


	ChronoGPU chrTiledCPU;
	chrTiledCPU.start();		// CPU method
	tiledMap(h_inTiledCPU.getPtr(), h_outTiledCPU.getPtr(), h_inTiledCPU.getWidth(), h_inTiledCPU.getHeight(), h_outTiledCPU.getWidth(), h_outTiledCPU.getHeight());			//void tiledMap(uint8_t *h_in, uint8_t *h_out,const int inMapWidth, const int inMapHeight , const int outMapWidth, const int outMapHeight)

	chrTiledCPU.stop();
	h_outTiledCPU.saveTo("img/Result/CPU/TiledCPU.ppm");
	

	const float timeComputeTiledCPU = chrTiledCPU.elapsedTime();
	std::cout << "-> Done : " << std::fixed << std::setprecision(2) << timeComputeTiledCPU << " ms" << std::endl
			  << std::endl;



	// ================================================================================================================

	// ================================================================================================================
	// GPU CUDA
	std::cout << "============================================" << std::endl;
	std::cout << "          Parallel version on GPU           " << std::endl;
	std::cout << "============================================" << std::endl;

	// data GPU

    Heightmap h_inGPU("img/input/1.input.ppm");			//1.input     limousin-full
    Heightmap h_outGPU(h_inGPU.getWidth(), h_inGPU.getHeight());
	std::vector<float> angle(h_inGPU.getWidth() * h_inGPU.getHeight());
	// data GPU
	std::cout << h_inGPU.getWidth() << " " << h_inGPU.getHeight() << std::endl;
	// GPU allocation
	ChronoGPU chrGPU;
	float moy;
	float timeAllocGPU;

	chrGPU.start();	
	carteGPUv1(h_inGPU.getPtr(), h_outGPU.getPtr(), h_inGPU.getWidth(), h_inGPU.getHeight(), Cx, Cy);
	//carteGPUv1(h_inGPU.getPtr(), h_outGPU.getPtr(), angle.data(), h_inGPU.getWidth(), h_inGPU.getHeight(), Cx, Cy);

	chrGPU.stop();
	timeAllocGPU = chrGPU.elapsedTime();
	h_outGPU.saveTo("img/Result/GPU/LimousinGPU.ppm");
	
	std::cout << "-> Done : " << std::fixed << std::setprecision(2) << timeAllocGPU << " ms en moyenne" << std::endl;


	//-----------------------------------------// TiledGPU



	Heightmap h_inTiledGPU("img/input/1.input.ppm");
    Heightmap h_outTiledGPU(TiledWidth, TiledHeight);

	TiledGPU(h_inTiledGPU.getPtr(), h_outTiledGPU.getPtr(), h_inTiledGPU.getWidth(), h_inTiledGPU.getHeight() , h_outTiledGPU.getWidth(), h_outTiledGPU.getHeight());

	h_outTiledGPU.saveTo("img/Result/GPU/TiledGPU.ppm");
	// ================================================================================================================

	std::cout << "============================================" << std::endl;
	std::cout << "              Checking results              " << std::endl;
	std::cout << "============================================" << std::endl;
	


	for (int i = 0; i < h_inCPU.getHeight(); i++)
	{
		for (int j = 0; j < h_inCPU.getWidth(); j++)
			{
				if (h_outCPU.getPixel(j, i) != h_outGPU.getPixel(j, i))
				{
					std::cout << "error on index (" << i << ", " << j << ")" << std::endl;
					std::cout << "value CPU : " << +h_outCPU.getPixel(j, i) << ", value GPU : " << +h_outGPU.getPixel(j, i) << std::endl;

					return EXIT_FAILURE;
				} 
			}
	}

	return EXIT_SUCCESS;
}