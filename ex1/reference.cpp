#include "reference.hpp"

void histogrameCPU( int *tab, int size, int* res ) {
	for(int i = 0; i<size; i++){
		int elem = tab[i];
		res[elem]++;
	}
}