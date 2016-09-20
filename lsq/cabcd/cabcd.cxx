#include <iostream>
#include <mkl.h>
#include "mpi.h"
#include <assert.h>
#include <stdlib.h>
#include <cstring>

#include "cabcd.h"
#include "util.h"

void cabcd(	double *X,	//input args
			int m,
			int n,
			double *y,
			double lambda,
			int s,
			int b,
			int maxit,
			double tol,
			int seed,
			int freq,
			double *w,
			MPI_Comm comm)	//output arg: allocated in function.
{
	int npes, rank;
	MPI_Comm_size(comm, &npes);
	MPI_Comm_rank(comm, &rank);

	double *alpha, *obj_err, *sol_err;
	double *del_w;

	double *G, *Xsamp;

	assert(0==Malloc_aligned(double, alpha, m, ALIGN));
	assert(0==Malloc_aligned(double, Xsamp, n*s*b, ALIGN));
	assert(0==Malloc_aligned(double, G, s*b*s*b, ALIGN));
	
	int localn = (rank < n%npes) ? (n/npes + 1) : n/npes;

	std::memset(alpha, 0, sizeof(double)*localn);
	std::memset(w, 0, sizeof(double)*m);

	char transa = 'T', transb = 'N';
	double alp = 1./n;
	double one = 1., zero = 0.;

	srand48(seed);

	double commst, commstp, commtot = 0.;
	double dgemmst, dgemmstp, dgemmtot = 0.;
	
	int iter = 0;
	int exit_flag = 0;
	
	int *index, *order;
	assert(0==Malloc_aligned(int, index, s*b, ALIGN));
	assert(0==Malloc_aligned(int, order, s*b, ALIGN));
	
	for(int i = 0; i < s*b; ++i){
		index[i] = lrand48()%m;
		order[i] = i/b;
	}


	//sampling tuning choices: randomly permute data matrix during I/O. after I/O. randomly select a column at a time.

	
}

int main (int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int npes, rank;
	int m, n;
	int *cnts, *cnts2, *displs, *displs2;
	double *X, *y, *localX, *localy;
	char *fname;

	
	double lambda, tol;
	int maxit, seed, freq, s, b;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);

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
	s = atoi(argv[9]);
	b = atoi(argv[10]);

	assert(0==Malloc_aligned(double, y, m, ALIGN));
	assert(0==Malloc_aligned(double, X, m*n, ALIGN));
	
	libsvmread(fname, m, n, X, m, y);
	
	//compute scatter offsets

	int flag = 1;
	cnts = Malloc(int, npes);
	cnts2 = Malloc(int, npes);
	displs = Malloc(int, npes);
	displs2 = Malloc(int, npes);

	staticLB_1d(m, n, npes, flag, cnts, displs, cnts2, displs2);

	assert(0==Malloc_aligned(double, localX, cnts[rank], ALIGN));
	assert(0==Malloc_aligned(double, localy, cnts2[rank], ALIGN));
	
	MPI_Scatterv(X, cnts, displs, MPI_DOUBLE, localX, cnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(y, cnts2, displs2, MPI_DOUBLE, localy, cnts2[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(cnts); free(displs);
	free(cnts2); free(displs2);

	if(rank == 0)
		free(X);

	double algst, algstp;
	double *w = Malloc(double, n);

	algst = MPI_Wtime();
	cabcd(localX, n, m, y, lambda, s, b, maxit, tol, seed, freq, w, comm);
	algstp = MPI_Wtime();

	free(localX); free(y);
	
	MPI_Finalize();
	return 0;
}
