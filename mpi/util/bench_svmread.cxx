#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "util.h"

int main(int argc, char* argv[]){
	
	MPI_Init(&argc, &argv);

	char* fname = argv[1];
	int m = atoi(argv[2]);
	int n = atoi(argv[3]);
	int flag = atoi(argv[4]);
	int p = atoi(argv[5]);

	double *A, *y;
	
	
	
	std::cout << "fname = "<< fname << " m = " << m << " n = " << n << std::endl;

	assert(0==Malloc_aligned(double, A, m*n, ALIGN));
	assert(0==Malloc_aligned(double, y, m, ALIGN));

	std::string lines = libsvmread(fname, m, n, A, m, y);
	/*
	for(int i = 0; i < 2; ++i)
		for(int j = 0; j < n; ++j)
			std::cout << A[i*n + j] << std::endl;
	*/
	std::vector<int> rowidx, colidx, rowoffsets;
	std::vector<double> y, vals;
	parse_lines_to_csr(lines, rowidx, colidx, vals, y, rowoffsets);
	
	MPI_Finalize();
	return 0;
	int *cnts, *displs, *cnts2, *displs2;
	cnts = Malloc(int, p);
	cnts2 = Malloc(int, p);
	displs = Malloc(int, p);
	displs2 = Malloc(int, p);
	staticLB_1d(m,n,p,flag, cnts, displs, cnts2, displs2);

	for(int i = 0; i < p; ++i){
		std::cout  << "cnts2[" << i << "] = "<< cnts2[i] << " ";
	}
	std::cout << std::endl;

	for(int i = 0; i < p; ++i){
		std::cout  << "displs2[" << i << "] = "<< displs2[i] << " ";
	}
	std::cout << std::endl;

	free(A); free(y);
}
