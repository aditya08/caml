#include <iostream>
#include <mkl.h>
#include "mpi.h"
#include <assert.h>
#include <stdlib.h>
#include <cstring>
#include <string.h>


#include "util.h"
#include "cabcd.h"

int main (int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	int npes, rank;
	int m, n;
	int *cnts, *cnts2, *displs, *displs2;
	double *X, *y, *localX, *localy;
	char *fname;

	
	double lambda, tol;
	int maxit, seed, freq, s, b;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm comm = MPI_COMM_WORLD;

	if(argc < 10)
	{
		if(rank == 0)
		{
			std::cout << "Bad args list!" << std::endl;
			std::cout << argv[0] << " [filename] [rows] [cols] [lambda] [maxit] [tol] [seed] [freq] [block size] [loop block size]" << std::endl;
		}

		MPI_Finalize();
		return -1;
	}

	fname = argv[1];
	m = atoi(argv[2]);
	n = atoi(argv[3]);

	lambda = atof(argv[4]);
	maxit = atoi(argv[5]);
	tol = atof(argv[6]);
	seed = atoi(argv[7]);
	freq = atoi(argv[8]);
	b = atoi(argv[9]);
	s = atoi(argv[10]);
	int niter = atoi(argv[11]);

	
	int flag = 0;
	cnts = Malloc(int, npes);
	cnts2 = Malloc(int, npes);
	displs = Malloc(int, npes);
	displs2 = Malloc(int, npes);

	staticLB_1d(m, n, npes, flag, cnts, displs, cnts2, displs2);

	assert(0==Malloc_aligned(double, localX, cnts[rank], ALIGN));
	assert(0==Malloc_aligned(double, localy, cnts2[rank], ALIGN));
	
	if(rank == 0 && strcmp(fname, "none") != 0){
		assert(0==Malloc_aligned(double, y, m, ALIGN));
		assert(0==Malloc_aligned(double, X, m*n, ALIGN));
		std::cout << "Reading file on rank 0" << std::endl;
		double iost = MPI_Wtime();
		libsvmread(fname, m, n, X, m, y);
		double iostp = MPI_Wtime();
		std::cout << "Finished reading file in " << iostp - iost << " seconds." << std::endl;
	}

	//compute scatter offsets

	if(strcmp(fname, "none") != 0){
		double scatterst = MPI_Wtime();
		MPI_Scatterv(X, cnts, displs, MPI_DOUBLE, localX, cnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(y, cnts2, displs2, MPI_DOUBLE, localy, cnts2[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		double scatterstp = MPI_Wtime();
		if(rank == 0){
			std::cout << "Finished Scatter of X and y in " << scatterstp - scatterst << " seconds." << std::endl;
			free(X); free(y);
		}
	}
	else{
		srand48(seed+rank);
		for(int i = 0; i < cnts[rank]; ++i)
			localX[i] = drand48();
		for(int i = 0; i < cnts2[rank]; ++i)
			localy[i] = drand48();
	}
	double algst, algstp;
	double *w;
	assert(0==Malloc_aligned(double, w, n, ALIGN));

	/*
	if(rank == 0){
	for(int i = 0; i < npes; ++i)
		std::cout << "cnts2[" << i << "] = " << cnts2[i];
	std::cout << std::endl;
	for(int i = 0; i < npes; ++i)
		std::cout << "cnts[" << i << "] = " << cnts[i];
	std::cout << std::endl;
	for(int i = 0; i < npes; ++i)
		std::cout << "displs2[" << i << "] = " << displs2[i];
	std::cout << std::endl;
	}
	*/

	if(rank == 0)
		std::cout << "Calling CA-BCD with " << n <<  "-by-" << m << " matrix X and s = " << s << std::endl;
	cabcd(localX, n, m, localy, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
	algst = MPI_Wtime();
	for(int i = 0; i < niter; ++i){
		cabcd(localX, n, m, localy, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
		
		/*
		if(rank == 0){
			std::cout << "w = ";
			for(int i = 0; i < n; ++i)
				printf("%.4f ", w[i]);
			std::cout << std::endl;
		}
		*/
	}
	algstp = MPI_Wtime();
		
	if(rank == 0)
		std::cout << std::endl << "Total CA-BCD time: " << (algstp - algst)/niter  << std::endl;
	/*
	if(rank == 0){
		std::cout << "w = ";
		for(int i = 0; i < n; ++i)
			std::cout << w[i] << " ";
		std::cout << std::endl;
	}
	*/

	free(localX); free(y);
	free(cnts); free(displs);
	free(cnts2); free(displs2);
	
	MPI_Finalize();
	return 0;
}
