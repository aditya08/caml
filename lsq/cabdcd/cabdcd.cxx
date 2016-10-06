#include <iostream>
#include <mkl.h>
#include "mpi.h"
#include <assert.h>
#include <stdlib.h>
#include <cstring>
#include <string.h>


#include "cabdcd.h"
#include "util.h"

void cabdcd(	double *X,	//input args
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
	double *del_a;

	double *G, *recvG, *Xsamp, *asamp, *ysamp;
	int incx = 1;
	int *index;
	int gram_size = s*b;
	int ngram = s*b*s*b;

	//std::cout << m << "-by-" << n << "local columns " << len << std::endl;

	assert(0==Malloc_aligned(double, alpha, n, ALIGN));
	assert(0==Malloc_aligned(double, Xsamp, len*gram_size, ALIGN));
	assert(0==Malloc_aligned(double, G, gram_size*(gram_size + 2), ALIGN));
	assert(0==Malloc_aligned(double, recvG, s*b*(s*b + 2), ALIGN));
	assert(0==Malloc_aligned(double, del_a, s*b, ALIGN));
	assert(0==Malloc_aligned(double, asamp, s*b, ALIGN));
	assert(0==Malloc_aligned(double, ysamp, s*b, ALIGN));
	assert(0==Malloc_aligned(int, index, s*b, ALIGN));

	
	//std::cout << "Initialized alpha and w to 0" << std::endl;
	memset(alpha, 0, sizeof(double)*n);
	memset(w, 0, sizeof(double)*len);

	char transa = 'N', transb = 'T', uplo = 'U';
	double alp = 1./(lambda*n*n);
	double bet = -1./n, gam = 1./n;
	double rho = 1./(lambda*n);
	double one = 1., zero = 0., neg = -1.;
	int info, nrhs = 1;
	int lGcols = b;

	srand48(seed);

	double commst, commstp, commagg = 0.;
	double gramst, gramstp, gramagg = 0.;
	double innerst, innerstp, inneragg = 0.;
	int iter = 0;
	int offset = 0;
	
	int cursamp, count;
	//std::cout << "local cols = " << len << std::endl;
	while(1){
		gramst = MPI_Wtime();
		for(int i = 0; i < s; ++i){
			cursamp = 0;
			count = 0;
			while(cursamp < b){
				if(((n-count)*drand48()) >= (b - cursamp))
					++count;
				else{
					index[cursamp + i*b] = count;
					//std::cout << "Index = " << count << std::endl;
					++count; ++cursamp;
				}
			}
		}

		//std::cout << "ysamp = ";
		for(int i = 0; i < s*b; ++i){
			//TODO: need to sample without replacement
			//if(rank ==0)
			//	std::cout << "index = " << index[i] << std::endl;
			dcopy(&len, X + index[i], &n, Xsamp + i, &gram_size);
			ysamp[i] = y[index[i]];
			//std::cout << ysamp[i] << " ";
			asamp[i] = alpha[index[i]];
		}
		//std::cout << std::endl;

		//for(int i =0; i < s*b*len; ++i)
		//	std::cout << "Xsamp[ " << i << "] = " << Xsamp[i] << std::endl;
		//std::cout << "Xsamp[0] = " << Xsamp[0] << " Xsamp[1] = " << Xsamp[1] << std::endl;  
		// Compute (s*b) x (s*b) Gram matrix
		
		//std::cout << "Calling DGEMM, lambda = " << lambda << " maxit = " << maxit  << " s = " << s << std::endl;
		
		dgemm(&transa, &transb, &gram_size, &gram_size, &len, &alp, Xsamp, &gram_size, Xsamp, &gram_size, &zero, G, &gram_size); 
		//std::cout << "dot product" << (1./n)*ddot(&len, Xsamp + 0, &gram_size, Xsamp + 1, &gram_size)+lambda << std::endl;



		// Compute y and alpha components of residual based on sampled rows.
		//std::cout << "Calling DGEMV" << std::endl;
		dgemv(&transa, &gram_size, &len, &bet, Xsamp, &gram_size, w, &incx, &zero, G + (s*b*s*b), &incx);
		//dgemv(&transa, &gram_size, &n, &alp, Xsamp, &gram_size, y, &incx, &zero, G + (s*b*(s*b+1)), &incx);
		gramstp = MPI_Wtime();
		gramagg += gramstp - gramst;

		// Reduce and Broadcast: Sum partial Gram and partial residual components.
		//std::cout << "Calling ALLREDUCE" << std::endl;
		commst = MPI_Wtime();
		MPI_Allreduce(G,recvG,s*b*(s*b+2), MPI_DOUBLE, MPI_SUM, comm);
		commstp = MPI_Wtime();
		commagg += commstp - commst;

		innerst = MPI_Wtime();
		for(int i =0; i < s*b; ++i)
				recvG[i + i*s*b] += 1./n;
		
		/*
		if(rank == 0){
			for(int i = 0; i < s*b; ++i){
					for(int j = 0; j < s*b; ++j)
						std::cout << recvG[j*s*b + i] << " ";
					std::cout << std::endl;
				}
				std::cout << std::endl;
		}
		MPI_Barrier(comm);
		*/
		/*
		if(rank == 0){
			std::cout << "reduced" << std::endl;
			for(int i = 0; i < s*b; ++i){
				for(int j = 0; j < s*b+2; ++j)
					std::cout << recvG[j*s*b + i] << " ";
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		*/

		/*
		 * Inner s-step loop
		 * Perfomed redundantly on all processors
		*/
		
		//combine residual updates into one vector

		daxpy(&gram_size, &gam, asamp, &incx, recvG + s*b*s*b, &incx);
		daxpy(&gram_size, &gam, ysamp, &incx, recvG + s*b*(s*b), &incx);
		
		dcopy(&gram_size, recvG + s*b*(s*b), &incx, del_a, &incx);
		
		/*
		if(rank == 0){
			std::cout << "residual on rank " << rank << " iter " << iter << std::endl;
			for(int i = 0; i < s*b; ++i){
					std::cout << del_a[i] << " ";
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		*/

		//compute solution to first (b) x (b) subproblem
		
		/*
		std::cout << "recvG[0] = " << recvG[0] << std::endl;
		std::cout << "before del_a = ";
		for(int j = 0; j < s*b; ++j){
			std::cout << del_a[j] << " ";
		}
		std::cout << std::endl;
		*/

		dpotrf(&uplo, &b, recvG, &gram_size, &info);
		assert(0==info);
		
		dpotrs(&uplo, &b, &nrhs, recvG, &gram_size, del_a, &b, &info);
		assert(0==info);
		for(int i = 0; i < b; ++i)
			alpha[index[i]] = alpha[index[i]] - del_a[i];
		iter++;
		
		if(rank == 0){
			std::cout << "del_a = ";
			for(int i = 0; i < b; ++i)
				std::cout << del_a[i] << " ";
			std::cout << std::endl;
		}
		//std::cout << "del_a = " << del_a[0] << std::endl; 
		innerstp = MPI_Wtime();
		inneragg += innerstp - innerst;

		if(iter == maxit){
			//gram_size = i*b;
			//dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_a, &incx, &one, alpha, &incx);
			dgemv(&transb, &b, &len, &rho, Xsamp, &gram_size, del_a, &incx, &one, w, &incx);
			free(alpha); free(Xsamp); free(G); free(recvG);
			free(index); free(del_a); free(ysamp); free(asamp);
			if(rank == 0){
				std::cout << "Outer loop computation time: " << gramagg << std::endl;
				std::cout << "Inner loop computation time: " << inneragg << std::endl;
				std::cout << "MPI_Allreduce time: " << commagg << std::endl;
			}
			/*
			for(int i = 0; i < m; ++i)
				std::cout << w[i] << " ";
			std::cout << std::endl;
			*/
			return;
		}
		//std::cout << "Iter count: " << iter << std::endl;
		//std::cout << "del_a before = " << del_a[iter] << std::endl; 
		if(rank == 0){
			std::cout << "del_a before = ";
			for(int i = b; i < s*b; ++i)
				std::cout << del_a[i] << " ";
			std::cout << std::endl;
		}

		for(int i = 1; i < s; ++i){
			
			// Compute residual based on previous subproblem solution
			innerst = MPI_Wtime();
			lGcols = i*b;
			dgemv(&transa, &b, &lGcols, &neg, recvG + i*b, &gram_size, del_a, &incx, &one, del_a + i*b, &incx);
			
			// Correct residual if any sampled row in current block appeared in any previous blocks
			for(int j = 0; j < i*b; ++j){
				for(int k = 0; k < b; ++k){
					if(index[j] == index[i*b + k])
						del_a[i*b + k] += (1./n)*del_a[j];
				}
			}

			// Compute solution to next (b) x (b) subproblem
			//std::cout << "recvG[" << i << "] = " << recvG[lGcols + s*lGcols] << std::endl;
			dpotrf(&uplo, &b, recvG + lGcols + s*b*lGcols, &gram_size, &info);
			assert(0==info);

			dpotrs(&uplo, &b, &nrhs, recvG + lGcols + s*b*lGcols, &gram_size, del_a + lGcols, &b, &info);
			assert(0==info);
			
			for(int j = 0; j < b; ++j)
				alpha[index[i*b + j]] = alpha[index[i*b + j]] - del_a[i*b + j];
			if(rank == 0){
				std::cout << "del_a = ";
				for(int k = 0; k < b; ++k)
					std::cout << del_a[i*b + k] << " ";
				std::cout << std::endl;
			}
			//std::cout << "del_a = " << del_a[i] << std::endl; 
			iter++;
			inneragg += MPI_Wtime() - innerst;
			if(iter == maxit){
				//dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_a, &incx, &one, alpha, &incx);
				lGcols = (i + 1)*b;
				dgemv(&transb, &lGcols, &len, &rho, Xsamp, &gram_size, del_a, &incx, &one, w, &incx);
				free(alpha); free(Xsamp); free(G); free(recvG);
				free(index); free(del_a); free(asamp); free(ysamp);
				if(rank == 0){
					std::cout << "Outer loop computation time: " << gramagg << std::endl;
					std::cout << "Inner loop computation time: " << inneragg << std::endl;
					std::cout << "MPI_Allreduce time: " << commagg << std::endl;
				}
				/*
				for(int i = 0; i < m; ++i)
					std::cout << w[i] << " ";
				std::cout << std::endl;
				*/
				return;
			}
			//std::cout << "Iter count: " << iter << std::endl;
		}
		/*
		std::cout << "after del_a = ";
		for(int j = 0; j < s*b; ++j){
			std::cout << del_a[j] << " ";
		}
		std::cout << std::endl;
		*/

		// Update w
		//std::cout << "w = ";
		//	std::cout << w[index[j]] << " ";
		//std::cout << std::endl;
		

		// Update local alpha
		gramst = MPI_Wtime();
		dgemv(&transb, &gram_size, &len, &rho, Xsamp, &gram_size, del_a, &incx, &one, w, &incx);
		
		memset(G, 0, sizeof(double)*gram_size*(gram_size+2));
		memset(recvG, 0, sizeof(double)*gram_size*(gram_size+2));
		memset(del_a, 0, sizeof(double)*gram_size);
		gramagg += MPI_Wtime() - gramst;
		/*
		 * End Inner s-step loop
		*/
		
	}

	
}

//sampling tuning choices: randomly permute data matrix during I/O. after I/O. randomly select a column at a time.
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

	
	int flag = 1;
	cnts = Malloc(int, npes);
	cnts2 = Malloc(int, npes);
	displs = Malloc(int, npes);
	displs2 = Malloc(int, npes);

	staticLB_1d(m, n, npes, flag, cnts, displs, cnts2, displs2);

	assert(0==Malloc_aligned(double, localX, cnts[rank], ALIGN));
	assert(0==Malloc_aligned(double, y, m, ALIGN));
	if(rank == 0 && strcmp(fname, "none") != 0){
		assert(0==Malloc_aligned(double, X, m*n, ALIGN));
		std::cout << "Reading file on rank 0" << std::endl;
		double iost = MPI_Wtime();
		libsvmread(fname, m, n, X, m, y);
		mkl_dimatcopy('C', 'T', n, m, 1.0, X, n, m); 
		double iostp = MPI_Wtime();
		std::cout << "Finished reading file in " << iostp - iost << " seconds." << std::endl;
	}
	//compute scatter offsets


	if(strcmp(fname, "none") != 0){
		
		//for(int i = 0; i < m*n; ++i)
		//	std::cout << X[i] << " ";
		//std::cout << std::endl;
		double scatterst = MPI_Wtime();
		
		for(int i = 0; i < npes; ++i){
			std::cout << cnts[i] << " ";
		}
		std::cout << std::endl;
		
		MPI_Scatterv(X, cnts, displs, MPI_DOUBLE, localX, cnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(y, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		std::cout << "cnts2[" << rank << "] = " << cnts2[rank] << std::endl;
		std::cout << y[0] << std::endl;
		double scatterstp = MPI_Wtime();
		if(rank == 0){
			std::cout << "Finished Scatter of X and y in " << scatterstp - scatterst << " seconds." << std::endl;
			//free(X); free(y);
		}
	}
	else{
		srand48(seed+rank);
		for(int i = 0; i < cnts[rank]; ++i)
			localX[i] = drand48();
		for(int i = 0; i < m; ++i)
			y[i] = drand48();
	}
	double algst, algstp;
	double *w;
	assert(0==Malloc_aligned(double, w, cnts2[rank], ALIGN));

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
		std::cout << "Calling CA-BDCD with " << n <<  "-by-" << m << " matrix X and s = " << s << std::endl;
	cabdcd(localX, n, m, y, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
	algst = MPI_Wtime();
	for(int i = 0; i < niter; ++i){
		cabdcd(localX, n, m, y, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
		
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
		
	if(rank == 0){
		std::cout << std::endl << "Total CA-BDCD time: " << (algstp - algst)/niter  << std::endl;
		free(X); free(y);
	}
	MPI_Barrier(comm);
	MPI_Barrier(comm);
	MPI_Barrier(comm);
	MPI_Barrier(comm);
	MPI_Barrier(comm);
	std::cout << "w = ";
	for(int i = 0; i < cnts2[rank]; ++i)
		std::cout << w[i] << " ";
	std::cout << std::endl;
	MPI_Barrier(comm);
	free(localX); free(w);
	free(cnts); free(displs);
	free(cnts2); free(displs2);
	
	MPI_Finalize();
	return 0;
}
