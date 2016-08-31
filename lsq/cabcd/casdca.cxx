#include <string>
#include <iostream>
#include <assert.h>
#include <string.h>
#include <math.h>

#include <mkl.h>


#include "libsvmread.h"
#include "casdca.h"

void casdca(int m, int n, double *X, int leny, double *y, double *w, double lambda, int maxit, double tol, int seed, int freq, int s, double *wopt, MPI_Comm comm){
	int npes, rank;
	MPI_Comm_size(comm, &npes);
	MPI_Comm_rank(comm, &rank);
	
	double *alpha;
	double *res, *resr, scal;
	double del_alpha, *gamma;
	int *index = (int *) malloc(sizeof(int)*s);
	int *colproc = (int *) malloc(sizeof(int)*s);
	int inc = 1;

	int localm = (rank < n%npes) ? (n/npes + 1) : n/npes;
	alpha = (double *) malloc(sizeof(double)*localm);
	memset(alpha, 0, sizeof(double)*localm);
	memset(w, 0, sizeof(double)*m);

	double *gram, *Xs, *Xr, *pexp, *win, *wout, *I;

	res = (double*) malloc(sizeof(double)*s);
	resr = (double*) malloc(sizeof(double)*s);
	gamma = (double*) malloc(sizeof(double)*s);
	pexp = (double*) malloc(sizeof(double)*s);
	assert(0==posix_memalign((void**)&Xs, ALIGN, sizeof(double)*(m+1)*s));
	assert(0==posix_memalign((void**)&Xr, ALIGN, sizeof(double)*(m+1)*s));
	assert(0==posix_memalign((void**)&gram, ALIGN, sizeof(double)*s*s));
	assert(0==posix_memalign((void**)&I, ALIGN, sizeof(double)*s*s));
	assert(0==posix_memalign((void**)&win, ALIGN, sizeof(double)*m));
	assert(0==posix_memalign((void**)&wout, ALIGN, sizeof(double)*m));
	

	int *displs = (int *) malloc(sizeof(int)*npes);
	
	char transa = 'T', transb = 'N'; 
	double alp = 1./(lambda*n*n), bet = 1./n;
	double one = 1., zero = 0.;
	//std::cout << "displ = " << displs[npes] <<  std::endl;

	srand(seed);

	double commst, commstp, commtot = 0.;
	double dgemmst, dgemmstp, dgemmtot = 0.;
	double memopst, memopstot = 0.;
	double inlpst, inlptot = 0.;
	double totst = MPI_Wtime();
	int iter = 0;
	int exit_flag = 0;
	
	int upbnd = (n/npes + 1)*(n%npes);

	while(1){
		/*Select s random columns.*/
		/*Compute which processors have those columns 
		 * and convert from global index to local index.*/
		for(int k = 0; k < s; ++k){
			index[k] = rand()%n;
			//if(rank == 0)
			//	std::cout << "index before: " << index[k] << " ";
			colproc[k] = (index[k] < upbnd) ? (index[k]/(n/npes + 1)) : ((index[k]-upbnd)/(n/npes) + (n%npes));
			index[k] -= (colproc[k] < n%npes) ? (((colproc[k])*(n/npes + 1))) : ((n%npes*(n/npes + 1)) + ((colproc[k] - n%npes)*(n/npes)));
			//if(rank == 0)
			//	std::cout << "colproc: " << colproc[k] << " index after: " << index[k] << std::endl;
		}

		/*If nrows smaller than nprocs, then use Allgatherv
		 * and redundantly compute Gram matrix, computation is cheap here!*/

		//TODO: Consider adding a blocksize parameter for better comp v. sync.
		if(m < npes){
			
		}

		/*Have enough rows for Alltoallv, so parallelize computation. 
		 * Each processor computes 1/p of Gram matrix and sum Allreduce.*/
		else{
		

		}

		memopst = MPI_Wtime();
		memset(Xs, 0, sizeof(double)*(m+1)*s);
		memset(Xr, 0, sizeof(double)*(m+1)*s);
		memset(I, 0, sizeof(double)*s*s);
		memset(pexp, 0, sizeof(double)*s);
		memset(res, 0, sizeof(double)*s);
		memset(resr, 0, sizeof(double)*s);
		memset(win, 0, sizeof(double)*m);
		memset(wout, 0, sizeof(double)*m);
		memset(gram, 0, sizeof(double)*s*s);
		memopstot += MPI_Wtime() - memopst;

		for(int k = 0; k < s; ++k){
			if(colproc[k] == rank){
				memcpy(Xs + k*m, X + index[k]*m, sizeof(double)*m);
				Xs[m*s+k] = ((-1./n)*ddot(&m, Xs + k*m, &inc, w, &inc)) + ((1./n)*alpha[index[k]]) + ((1./n)*y[index[k]]);
				//res[k] = ((-1./n)*ddot(&m, Xs + k*m, &inc, w, &inc)) + ((1./n)*alpha[index[k]]) + ((1./n)*y[index[k]]);
				//std::cout << "Res[" << k << "]: "<< res[k] << std::endl;
			}
			I[k + s*k] = 1.;
			gram[s*k + k] = 1.;
		}
		commst = MPI_Wtime();
		MPI_Allreduce(Xs, Xr, (m+1)*s, MPI_DOUBLE, MPI_SUM, comm);
		//MPI_Allreduce(res, resr, s, MPI_DOUBLE, MPI_SUM, comm);
		//memcpy(resr, Xr+s*m, sizeof(double)*s);
		
		commstp = MPI_Wtime();
		commtot += commstp - commst;
		/*
		if(rank == 0){
			for(int j = 0; j < m; ++j){
				for(int k = 0; k < s; ++k){
					std::cout << Xr[m*k + j] << " ";
				}
				std::cout << std::endl;
			}
		}
		*/

		/*redundantly compute Gram matrix*/
		dgemmst = MPI_Wtime();
		dgemm(&transa, &transb, &s, &s, &m, &alp, Xr, &m, Xr, &m, &bet, gram, &s);
		dgemmstp = MPI_Wtime();
		dgemmtot += dgemmstp - dgemmst;
		/*
		if(rank == 0){
			for(int j = 0; j < s; ++j){
				for(int k = 0; k < s; ++k){
					std::cout << gram[s*k + j] << " ";
				}
				std::cout << std::endl;
			}
		}
		*/

		for(int k = 0; k < s; ++k){
			gamma[k] = 1./(gram[k + s*k]);
			gram[s*k + k] = 1.;
			for(int j = s-1; k < j; --j){
				gram[s*j + k] = 0.;
			}
		}
		/*
		if(rank == 0){
			for(int j = 0; j < s; ++j){
				for(int k = 0; k < s; ++k){
					std::cout << gram[s*k + j] << " ";
				}
				std::cout << std::endl;
			}
		}
		*/
		for(int k = 0; k < s; ++k){
			inlpst = MPI_Wtime();
			dgemv(&transa, &s, &s, &one, I, &s, gram + k, &s, &zero, pexp , &inc);
			dcopy(&s, I + k, &s, pexp, &inc);
			//for(int j = 0; j < s; ++j)
			//	I[j*s + k] = pexp[j];
			if(colproc[k] == rank){
				/*
				std::cout << "pexp[" << k << "]: ";
				for(int j = 0; j < s; ++j)
					std::cout << pexp[j] << " ";
				std::cout << std::endl;
				*/

				del_alpha = -gamma[k]*ddot(&s, pexp, &inc, Xr+m*s, &inc);
				//std::cout << "iter: " << iter/s << " Delta alpha[" << k << "]: " << del_alpha << std::endl;
				alpha[index[k]] += del_alpha;
				scal = del_alpha/(lambda*n);
				daxpy(&m, &scal, X + (index[k]*m), &inc, win, &inc);
			}
			inlptot += MPI_Wtime() - inlpst;
			iter++;

			if(iter == maxit){
				commst = MPI_Wtime();
				MPI_Allreduce(win, wout, m, MPI_DOUBLE, MPI_SUM, comm);
				commstp = MPI_Wtime();
				commtot += commstp - commst;
				one = -1.;
				daxpy(&m, &one, wout, &inc, w, &inc);
				one = 1.;
				exit_flag = 1;
				break;			
			}
		}

		if(exit_flag)
			break;

		commst = MPI_Wtime();
		MPI_Allreduce(win, wout, m, MPI_DOUBLE, MPI_SUM, comm);
		commstp = MPI_Wtime();
		commtot += commstp - commst;

		one = -1.;
		daxpy(&m, &one, wout, &inc, w, &inc);
		one = 1.;


	}
	double totstp = MPI_Wtime();
	if(rank == 0){
		std::cout << "Communication time: " << commtot << std::endl;
		std::cout << "dgemm time: " << dgemmtot << std::endl;
		std::cout << "inner loop time: " << inlptot << std::endl;
		std::cout << "memops time: " << memopstot << std::endl;
		std::cout << "Total comp time: " << totstp - totst << std::endl;
	}
	
	free(alpha); 
	free(index); 
	free(colproc);
	
	free(win); free(wout); free(Xs); free(Xr);
	free(displs); free(I); free(gram);
	free(res); free(resr); free(gamma); free(pexp);
}

int main(int argc, char* argv[]){
	
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	int npes, rank;
	int m, n;
	double *A, *y, *localA, *localy;
	char *fname;

	double lambda, tol;
	int maxit, seed, freq, s;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	
	if(argc < 9){
		if(rank == 0){
			std::cout << "Bad args list!" << std::endl;
			std::cout << argv[0] << " [filename] [rows] [cols] [lambda] [maxit] [tol] [seed] [freq] [s]" << std::endl;
		}
		MPI_Finalize();
		return -1;
	}

	/*Init dataset specific params*/
	fname = argv[1];
	m = atoi(argv[2]);
	n = atoi(argv[3]);

	/*Init SDCA specific params*/
	lambda = atof(argv[4]);
	maxit = atoi(argv[5]);
	tol = atof(argv[6]);
	seed = atoi(argv[7]);
	freq = atoi(argv[8]);
	s = atoi(argv[9]);

	assert(0==posix_memalign((void**)&y, ALIGN, sizeof(double)*m));
	if(rank == 0){
		assert(0==posix_memalign((void**)&A, ALIGN, sizeof(double)*m*n));
		memset(A,0,sizeof(double)*m*n);
		libsvmread(fname, m, n, A, m, y);
		std::cout << "s = " << s << std::endl;
		/*
		for(int i = 0; i < 2; ++i)
			for(int j = 0; j < n; ++j)
			std::cout << A[i*n + j] << std::endl; 
		*/
	}

	/*Partition A into 1D-block rows and scatter*/
	int *cnts, *cnts2;
	int *displs, *displs2;
	cnts = (int*) malloc(sizeof(int)*npes);
	displs = (int*) malloc(sizeof(int)*npes);
	cnts2 = (int*) malloc(sizeof(int)*npes);
	displs2 = (int*) malloc(sizeof(int)*npes);

	for(int i = 0; i < m%npes; ++i){
		cnts[i] = (m/npes + 1)*n;
		cnts2[i] = (m/npes + 1);
		displs[i] = (i*(m/npes + 1))*n;
		displs2[i] = (i*(m/npes + 1));
		/*
		if(rank == 0){
			std::cout << cnts2[i] << std::endl;
			std::cout << displs2[i] << std::endl;
		}
		*/
	}
	for(int i = m%npes; i < npes; ++i){
		cnts[i] = (m/npes)*n;
		cnts2[i] = m/npes;
		displs[i] = ((m%npes)*(m/npes + 1) + (i - m%npes)*m/npes)*n;
		displs2[i] = ((m%npes)*(m/npes + 1) + (i - m%npes)*m/npes);
		/*
		if(rank == 0){
			std::cout << cnts2[i] << std::endl;
			std::cout << displs2[i] << std::endl;
		}
		*/
	}

	assert(0==posix_memalign((void**)&localA, ALIGN, sizeof(double)*cnts[rank]*n));
	assert(0==posix_memalign((void**)&localy, ALIGN, sizeof(double)*cnts[rank]));
	MPI_Scatterv(A, cnts, displs, MPI_DOUBLE, localA, cnts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(y, cnts2, displs2, MPI_DOUBLE, localy, cnts2[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	free(cnts); free(displs);
	free(cnts2); free(displs2);
	/*
	if(rank == 1){
	std::cout << "Before transpose: " << std::endl;
	for(int i = 0; i < 2; ++i)
		for(int j = 0; j < n; ++j)
		std::cout << localy[j + i*n] << std::endl; 
	}
	*/

	if(rank == 0)
		free(A);
	
	/*Start sdca iterations*/
	double *w;
	assert(0==posix_memalign((void**)&w, ALIGN, sizeof(double)*n));
	
	double tst, tstp;
	tst = MPI_Wtime();
	casdca(n, m, localA, m, localy, w, lambda, maxit, tol, seed, freq, s, NULL, comm);
	tstp = MPI_Wtime();

	if(rank == 0){
		for(int k = 0; k < n; ++k)
			std::cout << w[k] << " ";
		std::cout << std::endl;

		std::cout << "Total running time: " << tstp - tst << " seconds" << std::endl;
	}

	/*Clean-up*/
	free(localA); free(y);
	MPI_Finalize();
	return 0;
}
