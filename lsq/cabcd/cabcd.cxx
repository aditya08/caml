#include <mlk.h>
#include "mpi.h"

#include "cabcd.h"
#include "util.h"

void cabcd(	double *X,	//input args
			int m,
			int n,
			double *y,
			double *lambda,
			int s,
			int b,
			int maxit,
			double tol,
			int seed,
			int freq,
			double **w,
			MPI_Comm comm)	//output arg: allocated in function.
{
	double *alpha, *obj_err, *sol_err, 
	double del_w;

	double *G, *Xsamp;

	//sampling tuning choices: randomly permute data matrix during I/O. after I/O. randomly select a column at a time.

	
}

int main (int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int npes, rank;
	int m, n;
	int *cnts, *cnts2; *displs; *displs2;
	double *X, *y, *localX, *localy;
	char *fname;

	double lambda, tol;
	int maxit, seed, freq, s;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);

	if(argc < 10)
	{
		if(rank == 0)
		{
			std::cout << "Bad args list!" << std:endl;
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
	assert(0==Malloc_aligned(double, A, m*n, ALIGN));
	
	libsvmread(fname, m, n, A, m, y);
	
	//compute scatter offsets

	staticLB_1d(m, n, npes, &cnts, &displs, &cnts2, &displs2);

	assert(0==Malloc_aligned(double, localX, cnts[rank]*n, ALIGN));
	assert(0==Malloc_aligned(double, localy, cnts[rank], ALIGN));
	
	MPI_Scatterv(X, cnts, displs, MPI_DOUBLE, localA, cnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(y, cnts2, displs2, MPI_DOUBLE, localy, cnts2[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(cnts); free(displs);
	free(cnts2); free(displs2);

	if(rank == 0)
		free(X);

	double algst, algstp;
	double *w;

	algst = MPI_Wtime();
	cabcd(localX, y, lambda, s, b, maxit, tol, seed, freq, &w, comm);
	algstp = MPI_Wtime();

	free(localX); free(y);
	
	MPI_Finalize();
	return 0;
}
