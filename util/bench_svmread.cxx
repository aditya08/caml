#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"

int main(int argc, char* argv[]){
	char* fname = argv[1];
	int m = atoi(argv[2]);
	int n = atoi(argv[3]);

	double *A, *y;

	std::cout << "fname = "<< fname << " m = " << m << " n = " << n << std::endl;
	
	assert(0==Malloc_aligned(double, A, m*n, ALIGN));
	assert(0==Malloc_aligned(double, y, m, ALIGN));

	libsvmread(fname, m, n, A, m, y);
	
	for(int i = 0; i < 2; ++i)
		for(int j = 0; j < n; ++j)
			std::cout << A[i*n + j] << std::endl;

	free(A); free(y);
}
