#include <iostream>
#include <mkl.h>
#include "mpi.h"
#include <assert.h>
#include <stdlib.h>
#include <cstring>
#include <iomanip>
#include <string.h>


#include "cabdcd.h"
#include "util.h"

void cabdcd(				std::vector<int> &rowidx,
							std::vector<int> &colidx,
							std::vector<double> &vals,
							int m,
							int n,
							std::vector<double> &y,
							int len,
							double lambda,
							int s,
							int b,
							int maxit,
							double tol,
							int seed,
							int freq,
							double *w,
							MPI_Comm comm)
{
	int npes, rank;
	MPI_Comm_size(comm, &npes);
	MPI_Comm_rank(comm, &rank);

//	std::cout << " y.size() " << y.size() << std::endl;
	double *alpha, *sampres, *sampres_sum, *res,  *obj_err, *sol_err;
	double *del_a;

	double *G, *recvG, *Xsamp, *asamp, *ysamp;
	int incx = 1;
	int *index;
	int gram_size = s*b;
	int ngram = s*b*s*b;

	std::cout << std::setprecision(4) << std::fixed;
	//std::cout << m << "-by-" << n << "local columns " << len << std::endl;

	assert(0==Malloc_aligned(double, alpha, n, ALIGN));
	assert(0==Malloc_aligned(double, sampres, n, ALIGN));
	assert(0==Malloc_aligned(double, sampres_sum, n, ALIGN));
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
	int rescnt = 0;
	double resnrm;

	char matdesc[6];
	matdesc[0] = 'G'; matdesc[3] = 'F';

	srand48(seed);

	double commst, commstp, commagg = 0.;
	double gramst, gramstp, gramagg = 0.;
	double innerst, innerstp, inneragg = 0.;
	int iter = 0;
	int offset = 0;

	int cursamp, count;
	std::vector<int> samprowidx(rowidx.size(), 1);
	std::vector<int> sampcolidx;
	std::vector<double> sampvals;
	int cidx = 0, rnnz = 0;
	double tval = 0.;
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
		//std::cout << "y[index] " << y[index[0]] << std::endl;
		//int col = 0;
		//for(int i = 0; i < colidx.size(); ++i){
		//	col = (col > colidx[i]) ? col : colidx[i];
		//}
		//std::cout << " largest colidx = " << col << std::endl;
		//std::cout << " largest rowidx = " << rowidx.size()-1 << std::endl;

		for(int i = 1; i < rowidx.size(); ++i){
			samprowidx[i] = samprowidx[i-1];
			//std::cout << samprowidx.size()<< std::endl;
			for(int k = 0; k < gram_size; ++k){
				for(int j = rowidx[i-1]-1; j < rowidx[i]-1; ++j){
					cidx = colidx[j];
					tval = vals[j];
					if(cidx == index[k]+1){
						//std::cout << "currently cidx = " << cidx << std::endl;
						sampcolidx.push_back(k+1);
						sampvals.push_back(tval);
						//std::cout << "Incrementing samprowidx[" << i << "] to " << samprowidx[i] + 1 << std::endl;
						samprowidx[i]++;
					}
				}
			}
		}
		/*
		if(rank == 144 || rank == 146 || rank == 155 || rank == 158 ){
			std::cout << "rank = " << rank << " len = " << len << " sampvals.size() " << sampvals.size() << " sampcolidx.size() " << sampcolidx.size() << " samprowidx.size() " << samprowidx.size() << std::endl; 

			std::cout << "index = " << index[0] << " y.size() " << y.size() << std::endl;
		}
		*/
		//std::cout << "ysamp = ";
		for(int i = 0; i < s*b; ++i){
			//TODO: need to sample without replacement
			//if(rank ==0)
			//	std::cout << "index = " << index[i] << std::endl;
			//dcopy(&len, X + index[i], &n, Xsamp + i, &gram_size);
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

		//dgemm(&transa, &transb, &gram_size, &gram_size, &len, &alp, Xsamp, &gram_size, Xsamp, &gram_size, &zero, G, &gram_size);
		//std::cout << "dot product" << (1./n)*ddot(&len, Xsamp + 0, &gram_size, Xsamp + 1, &gram_size)+lambda << std::endl;

		//std::cout << "rank = " << rank << " finished selecting block and creating sampled matrix. ncols = " << len << " m = " << m << " n = " << n<< " sampcolidx.size() = " << sampcolidx.size() << std::endl;
		//for(int i = 0; i < sampcolidx.size(); ++i){
		//	std::cout << sampcolidx[i] << std::endl;
		//}
		
		mkl_dcsrmultd(&transb, &len, &gram_size, &gram_size, &sampvals[0], &sampcolidx[0], &samprowidx[0],  &sampvals[0], &sampcolidx[0], &samprowidx[0], G, &gram_size);
		//std::cout << "G[0]" << G[0] << std::endl;
		dscal(&ngram, &alp, G, &incx);
		

		// Compute y and alpha components of residual based on sampled rows.
		//std::cout << "Calling DGEMV" << std::endl;
		mkl_dcsrmv(&transb, &len, &gram_size, &bet, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], w, &zero, G+(s*b*s*b));

		//dgemv(&transa, &gram_size, &len, &bet, Xsamp, &gram_size, w, &incx, &zero, G + (s*b*s*b), &incx);
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

		//std::cout << "recvG[0]" << recvG[0] << std::endl;
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

		//std::cout << "asamp[0] " << asamp[0] << std::endl;
		//std::cout << "ysamp[0] " << ysamp[0] << std::endl;
		daxpy(&gram_size, &gam, asamp, &incx, recvG + s*b*s*b, &incx);
		daxpy(&gram_size, &gam, ysamp, &incx, recvG + s*b*(s*b), &incx);

		dcopy(&gram_size, recvG + s*b*(s*b), &incx, del_a, &incx);
		//std::cout << "del_a[0] " << del_a[0] << std::endl;
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

		/*
		if(rank == 0){
			std::cout << "del_a = ";
			for(int i = 0; i < b; ++i)
				std::cout << del_a[i] << " ";
			std::cout << std::endl;
		}
		*/
		//std::cout << "del_a = " << del_a[0] << std::endl;
		innerstp = MPI_Wtime();
		inneragg += innerstp - innerst;

		if(iter == maxit){
			//gram_size = i*b;
			//dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_a, &incx, &one, alpha, &incx);
			//dgemv(&transb, &b, &len, &rho, Xsamp, &gram_size, del_a, &incx, &one, w, &incx);
			mkl_dcsrmv(&transa, &len, &gram_size, &rho, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], del_a, &one, w);
			free(alpha); free(Xsamp); free(G); free(recvG);
			free(index); free(del_a); free(ysamp); free(asamp);
			if(rank == 0){
				std::cout << "Outer loop computation time: " << 1000*(gramagg)<< " ms"  << std::endl;
				std::cout << "Inner loop computation time: " << 1000*(inneragg)<< " ms"  << std::endl;
				std::cout << "MPI_Allreduce time: " << 1000*(commagg) << " ms" << std::endl;
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
		/*
		if(rank == 0){
			std::cout << "del_a before = ";
			for(int i = b; i < s*b; ++i)
				std::cout << del_a[i] << " ";
			std::cout << std::endl;
		}
		*/

		for(int i = 1; i < s; ++i){

			// Compute residual based on previous subproblem solution
			innerst = MPI_Wtime();
			lGcols = i*b;
			dgemv(&transa, &b, &lGcols, &neg, recvG + i*b, &gram_size, del_a, &incx, &one, del_a + i*b, &incx);

			// Correct residual if any sampled row in current block appeared in any previous blocks
			for(int j = 0; j < i*b; ++j){
				for(int k = 0; k < b; ++k){
					if(index[j] == index[i*b + k])
						del_a[i*b + k] -= (1./n)*del_a[j];
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
			/*
			if(rank == 0){
				std::cout << "del_a = ";
				for(int k = 0; k < b; ++k)
					std::cout << del_a[i*b + k] << " ";
				std::cout << std::endl;
			}
			*/
			//std::cout << "del_a = " << del_a[i] << std::endl;
			iter++;
			inneragg += MPI_Wtime() - innerst;
			if(iter == maxit){
				//dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_a, &incx, &one, alpha, &incx);
				lGcols = (i + 1)*b;
				//dgemv(&transb, &lGcols, &len, &rho, Xsamp, &gram_size, del_a, &incx, &one, w, &incx);
				mkl_dcsrmv(&transa, &len, &gram_size, &rho, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], del_a, &one, w);
				free(alpha); free(Xsamp); free(G); free(recvG);
				free(index); free(del_a); free(asamp); free(ysamp);
				if(rank == 0){
					std::cout << "Outer loop computation time: " << 1000*(gramagg)<< " ms"  << std::endl;
					std::cout << "Inner loop computation time: " << 1000*(inneragg)<< " ms"  << std::endl;
					std::cout << "MPI_Allreduce time: " << 1000*(commagg) << " ms" << std::endl;
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
		//dgemv(&transb, &gram_size, &len, &rho, Xsamp, &gram_size, del_a, &incx, &one, w, &incx);
		mkl_dcsrmv(&transa, &len, &gram_size, &rho, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], del_a, &one, w);
		
		if(iter/freq == rescnt){
			/*
			std::cout << "len = " << len << std::endl;
			std::cout << "nrows = " << rowidx.size()-1 << std::endl;
			std::cout << "ncols = " << y.size() << std::endl;
			std::cout << "colidx.size() = " << colidx.size() << std::endl;
			std::cout << "vals.size() = " << vals.size() << std::endl;
			for(int i = 0; i < rowidx.size(); ++i)
				std::cout << rowidx[i] << ' ';
			std::cout << std::endl;

			for(int i = 0; i < colidx.size(); ++i)
				std::cout << colidx[i] << ' ';
			std::cout << std::endl;
			
			for(int i = 0; i < vals.size(); ++i)
				std::cout << vals[i] << ' ';
			std::cout << std::endl;
			*/
			mkl_dcsrmv(&transb, &len, &n, &bet, matdesc, &vals[0], &colidx[0], &rowidx[0], &rowidx[1], &w[0], &zero, &sampres[0]);
			//std::cout << "sampres[1] = " << sampres[0] << std::endl;
			//std::cout << "sampres[n] = " << sampres[n-1] << std::endl;
			//free(alpha);
			MPI_Allreduce(sampres, sampres_sum, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			daxpy(&n, &gam, alpha, &incx, sampres_sum, &incx);
			daxpy(&n, &gam, &y[0], &incx, sampres_sum, &incx);

			resnrm = dnrm2(&n, sampres_sum, &incx);
			//if(rank == 0)
			//	std::cout << "current dual residual: " << resnrm << std::endl;
			rescnt++;
			if(resnrm <= tol){
				free(alpha); free(Xsamp); free(G); free(recvG);
				free(index); free(del_a); free(ysamp); free(asamp);
				samprowidx.clear(); sampcolidx.clear(); sampvals.clear();
				
				if(rank == 0){
					std::cout << "CA-BDCD converged with dual residual: " << resnrm << std::setprecision(4) << " At outer iteration: " << iter/s << std::endl;
					std::cout << "Outer loop computation time: " << gramagg*1000 << " ms" << std::endl;
					std::cout << "Inner loop computation time: " << inneragg*1000 << " ms" << std::endl;
					std::cout << "MPI_Allreduce time: " << commagg*1000 << " ms" << std::endl;
				}
				return;
			}
		}
		
		gramagg += MPI_Wtime() - gramst;

		sampcolidx.clear(); sampvals.clear();
		samprowidx[0] = 1;
		memset(G, 0, sizeof(double)*gram_size*(gram_size+2));
		memset(recvG, 0, sizeof(double)*gram_size*(gram_size+2));
		memset(del_a, 0, sizeof(double)*gram_size);

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


	std::string lines = libsvmread(fname, m, n);

	std::vector<int> rowidx, colidx;
	std::vector<double> y, vals;

	int dual_method = 0;
	parse_lines_to_csr(lines, rowidx, colidx, vals, y, dual_method, m , n);

	//for(int i = 0; i < y.size(); ++i)
	//	std::cout << y[i] << std::endl;
	//std::cout << std::endl;

	std::vector<int> ylen(npes,0);
	std::vector<int> ydispl(npes,0);
	ylen[rank] = y.size();
	int ysize = y.size();
	//std::cout << "y.size() " << y.size() << std::endl;
	MPI_Allgather(&ysize, 1, MPI_INT, &ylen[0], 1, MPI_INT, MPI_COMM_WORLD);
	
	ysize = (ylen[0] > n) ? n: ylen[0];
	ylen[0] = ysize;

	//if(rank == 0)
	//std::cout << "ylen.back() " << ylen[0] << std::endl;
	for(int i = 1; i < npes; ++i){
		if(ysize < n){
			ydispl[i] = ydispl[i-1] + ylen[i-1];
			ysize += ylen[i];
			if(ysize > n){
				ylen[i] -= (ysize - n);
			}
		}
		else{
			ylen[i] = 0;
			ydispl[i] = ydispl[i-1];
		}
		//if(rank == 0)
		//std::cout << "ylen.back() " << ylen[i] << std::endl;
	}

	std::vector<double> gathered_y(n, 0.);
	MPI_Allgatherv(&y[0], ylen[rank], MPI_DOUBLE, &gathered_y[0], &ylen[0], &ydispl[0], MPI_DOUBLE, MPI_COMM_WORLD);

	y = gathered_y;
	//for(int i = 0; i < y.size(); ++i)
	//	std::cout << y[i] << std::endl;
	//std::cout << std::endl;
	//std::cout << "ylen[0] " << ylen[rank] << std::endl;
	gathered_y.clear(); ydispl.clear(); ylen.clear();

	//std::cout << "y.size() " << y.size() << std::endl;

	double algst, algstp;
	double *w;
	int ncols = rowidx.size()-1;
	assert(0==Malloc_aligned(double, w, ncols, ALIGN));
		std::cout << std::setprecision(4) << std::fixed;
		/*
		for(int i = 0; i < m*n; ++i){
			std::cout << colidx[i] << ':' << vals[i] << ' ';
		}
		std::cout << std::endl;
		*/

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


	s = 1;
	b = 1;
	for(int k = 0; k < 2; ++k){
		if(b > n)
			continue;
		for(int j = 0; j < 6; ++j){
			if(rank == 0){
				std::cout << std::endl << std::endl;
				std::cout << "s = " << s << ", " << "b = " << b << std::endl;
			}
			//cabcd(localX, n, m, localy, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
			//cabdcd(rowidx, colidx, vals, m, n, y, ncols, lambda, s, b, maxit, tol, seed, freq, w, comm);
			algst = MPI_Wtime();
			for(int i = 0; i < niter; ++i){
				//cabcd(localX, n, m, localy, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
				cabdcd(rowidx, colidx, vals, m, n, y, ncols, lambda, s, b, maxit, tol, seed, freq/b, w, comm);
			}
				/*if(rank == 0){
					std::cout << "w = ";
					for(int i = 0; i < ncols; ++i)
						printf("%.4f ", w[i]);
					std::cout << std::endl;
				}*/
			algstp = MPI_Wtime();
			double algmaxt = 1000*(algstp - algst)/niter;
			double algmax;
			MPI_Reduce(&algmaxt, &algmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			if(rank == 0)
				std::cout << std::endl << "Total CA-BDCD time: " << algmax << " ms" << std::endl;
			s *= 2;
		}
		s = 1;
		b *= 8;
	}
	/*
	std::cout << "rank = " << rank <<  " w = ";
	for(int i = 0; i < ncols; ++i)
		printf("%.4f ", w[i]);
	std::cout << std::endl;
	*/

	free(w); rowidx.clear(); colidx.clear(); vals.clear(); y.clear();

	MPI_Finalize();
	return 0;
}
