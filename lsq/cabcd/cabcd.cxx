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
			int len,
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

	double *alpha, *res,  *obj_err, *sol_err;
	double *del_w;

	double *G, *recvG, *Xsamp, *wsamp;
	int incx = 1;

	assert(0==Malloc_aligned(double, alpha, len, ALIGN));
	assert(0==Malloc_aligned(double, Xsamp, n*s*b, ALIGN));
	assert(0==Malloc_aligned(double, G, s*b*(s*b + 2), ALIGN));
	assert(0==Malloc_aligned(double, recvG, s*b*(s*b + 2), ALIGN));
	assert(0==Malloc_aligned(int, index, s*b, ALIGN));
	assert(0==Malloc_aligned(int, del_w, s*b, ALIGN));
	assert(0==Malloc_aligned(int, wsamp, s*b, ALIGN));
	
	std::memset(alpha, 0, sizeof(double)*len);
	std::memset(w, 0, sizeof(double)*m);

	char transa = 'N', transb = 'T', uplo = 'U';
	double alp = 1./n;
	double one = 1., zero = 0., neg = -1.;
	int info, nrhs = 1;
	int lGcols = b;

	srand48(seed);

	double commst, commstp, commtot = 0.;
	double dgemmst, dgemmstp, dgemmtot = 0.;
	
	int iter = 0;
	int exit_flag = 0;
	
	int *index, *order;
	int gram_size = s*b;

	
	while(1){
		for(int i = 0; i < s*b; ++i){
			//TODO: need to sample without replacement
			index[i] = lrand48()%m;
			dcopy(&n, X + index[i], &m, Xsamp + i, &m);
			G[i + i*s*b] = 1.;
			wsamp[i] = w[index[i]];
		}
		// Compute (s*b) x (s*b) Gram matrix
		dgemm(&transa, &transb, &gram_size, &gram_size, &len, &alp, Xsamp, &gram_size, Xsamp, &len, &lambda, G, &gram_size); 
		
		// Compute y and alpha components of residual based on sampled rows.
		dgemv(&transa, &gram_size, &len, &alp, Xsamp, &gram_size, alpha, &incx, &zero, G + (s*b*s*b), &incx);
		dgemv(&transa, &gram_size, &len, &alp, Xsamp, &gram_size, y, &incx, &zero, G + (s*b*(s*b+1)), &incx);

		// Reduce and Broadcast: Sum partial Gram and partial residual components.
		MPI_Allreduce(G,recvG,s*b*(s*b+2), MPI_DOUBLE, MPI_SUM, comm);
		
		/*
		 * Inner s-step loop
		 * Perfomed redundantly on all processors
		*/
		
		//combine residual updates into one vector
		daxpy(&gram_size, &lambda, w, &incx, G + s*b*s*b, &incx);
		daxpy(&gram_size, &neg, G + s*b*s*b, &incx, G + s*b*(s*b+1), &incx);

		//compute solution to first (b) x (b) subproblem
		dpotrs(&uplo, &b, &nrhs, G, &b, del_w, &b, &info);
		assert(0==info);
		for(int i = 1; i < s; ++i){
			
			// Compute residual based on previous subproblem solution
			lGcols = i*b;
			dgemv(&transa, &b, &lGcols, &neg, G + i*b, &lGcols, del_w, &incx, &zero, del_w + i*b, &incx);
			
			// Correct residual if any sampled row in current block appeared in any previous blocks
			for(int j = 0; j < i*b; ++j){
				for(int k = 0; k < b; ++k){
					if(index[j] == index[i*b + k])
						del_w[i*b + k/b] -= lambda*del_w[j];
				}
			}

			// Compute solution to next (b) x (b) subproblem
			dpotrs(&uplo, &b, &nrhs, G + i*b + i*s*b, &b, del_w + i*b, &b, &info);
			assert(0==info);
			
			// Update w
			for(int j = 0; j < b; ++j)
				w[index[i*b + j]] = w[index[i*b+j]] + del_w[i*b + j];

			iter+=1;

			if(iter == maxit){
				//gram_size = i*b;
				//dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_w, &incx, &one, alpha, &incx);
				free(alpha); free(Xsamp); free(G); free(recvG);
				free(index); free(del_w); free(wsamp);
				return;
			}
		}
		// Update local alpha
		dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_w, &incx, &one, alpha, &incx);


		/*
		 * End Inner s-step loop
		*/
		
	}

	
}

//sampling tuning choices: randomly permute data matrix during I/O. after I/O. randomly select a column at a time.
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
	cabcd(localX, n, m, y, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
	algstp = MPI_Wtime();

	free(localX); free(y);
	
	MPI_Finalize();
	return 0;
}
