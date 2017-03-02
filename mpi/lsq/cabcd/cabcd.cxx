#include <iostream>
#include <mkl.h>
#include "mpi.h"
#include <assert.h>
#include <stdlib.h>
#include <cstring>
#include <string.h>
#include <iomanip>

#include "cabcd.h"
#include "util.h"

void cabcd(	std::vector<int> &rowidx,
						std::vector<int> &colidx,
						std::vector<double> &vals,	//input args
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
						MPI_Comm comm)	//output arg: allocated in function.
{
	int npes, rank;
	MPI_Comm_size(comm, &npes);
	MPI_Comm_rank(comm, &rank);


	double *alpha, *res,  *obj_err, *sol_err;
	double *del_w;
	double ctol;
	double *G, *recvG, *Xsamp, *wsamp, *sampres;
	int incx = 1;
	int *index;
	int gram_size = s*b;
	int ngram = s*b*s*b;
	//if(rank == 0)
	//std::cout << m << "-by-" << n << "local columns " << len << std::endl;

	//if(y.size() > rowidx.size()-1){
	//	std::cout << "rank = " << rank << " ";
	//	for(int i = 0; i < y.size(); ++i)
	//		std::cout << y[i] << " ";
	//	std::cout << std::endl;
	//}

	assert(0==Malloc_aligned(double, alpha, len, ALIGN));
	assert(0==Malloc_aligned(double, G, gram_size*(gram_size + 2), ALIGN));
	assert(0==Malloc_aligned(double, recvG, s*b*(s*b + 2), ALIGN));
	assert(0==Malloc_aligned(double, del_w, s*b, ALIGN));
	assert(0==Malloc_aligned(double, wsamp, s*b, ALIGN));
	assert(0==Malloc_aligned(int, index, s*b, ALIGN));
	assert(0==Malloc_aligned(double, sampres, n, ALIGN));

	for(int i = 0; i < n; ++i)
		sampres[i] = 1.;
	//if(rank == 0)
	//std::cout << "Initialized alpha and w to 0" << std::endl;
	memset(alpha, 0, sizeof(double)*len);
	memset(w, 0, sizeof(double)*n);

	char transa = 'N', transb = 'T', uplo = 'U';
	double alp = 1./m;
	double one = 1., zero = 0., neg = -1.;
	int info, nrhs = 1;
	int lGcols = b;
	char matdesc[6];
	matdesc[0] = 'G'; matdesc[3] = 'F';
	srand48(seed);

	double commst, commstp, commagg = 0.;
	double gramst, gramstp, gramagg = 0.;
	double innerst, innerstp, inneragg = 0.;
	int iter = 0;
	int offset = 0;

	int convcnt = 0;
	int conviter = 0;
	int convthresh = (n%b == 0) ? n/b : n/b + 1;

	int cursamp, count;
	std::vector<int> samprowidx(rowidx.size(), 1);
	std::vector<int> sampcolidx;
	std::vector<double> sampvals;
	int cidx = 0, rnnz = 0;
	double tval = 0.;

	//std::vector<double> sampres(n, 1.);
	double grammax, innermax, commmax;
	//std::cout << "local cols = " << len << std::endl;
	while(1){
		//std::cout << iter << std::endl;
		gramst = MPI_Wtime();
		for(int i = 0; i < s; ++i){
			cursamp = 0;
			count = 0;
			while(cursamp < b){
				if(((n-count)*drand48()) >= (b - cursamp))
					++count;
				else{
					index[cursamp + i*b] = count;
					//if(rank == 0)
						//std::cout << count << ' ';
					++count; ++cursamp;
				}
			}
			//std::cout << std::endl;
			/*
			while(cursamp < b){
				if(((m-count)*drand48()) >= (b - cursamp))
					++count;
				else{
					index[cursamp + i*b] = count;
					//std::cout << count << std::endl;
					++count; ++cursamp;
				}
			}
			*/
		}
	//if(rank == 0)
	//std::cout << "Calling Sparse DGEMM, lambda = " << lambda << " iter = " << iter  << " s = " << s  << " b = " << b<< std::endl;


	/*for(int i = 0; i < colidx.size(); ++i)
		std::cout << colidx[i] << ' ';
	std::cout << std::endl;*/

	for(int i = 1; i < rowidx.size(); ++i){
		samprowidx[i] = samprowidx[i-1];
		//std::cout << samprowidx.size()<< std::endl;
		for(int k = 0; k < gram_size; ++k){
			for(int j = rowidx[i-1]-1; j < rowidx[i]-1; ++j){
				cidx = colidx[j];
				tval = vals[j];
				if(cidx == index[k]+1){
					//std::cout << "currently i = " << i << std::endl;
					sampcolidx.push_back(k+1);
					sampvals.push_back(tval);
					//std::cout << "Incrementing samprowidx[" << i << "] to " << samprowidx[i] + 1 << std::endl;
					samprowidx[i]++;
				}
			}
		}
	}

	/*
	for(int i = 0; i < sampcolidx.size(); ++i)
		std::cout << sampcolidx[i] << ' ';
	std::cout << std::endl;
	for(int i = 0; i < sampvals.size(); ++i)
		std::cout << std::setprecision(4) << std::fixed <<sampvals[i] << ' ';
	std::cout << std::endl;
	for(int i = 0; i < samprowidx.size(); ++i)
		std::cout << samprowidx[i] << ' ';
	std::cout << std::endl;
	*/
	//if(rank == 0){
	//	std::cout << "sampcolidx length = " << sampcolidx.size() << std::endl;
	//	std::cout << "norws = " << samprowidx.size() - 1 << std::endl;
	//}
	mkl_dcsrmultd(&transb, &len, &gram_size, &gram_size, &sampvals[0], &sampcolidx[0], &samprowidx[0],  &sampvals[0], &sampcolidx[0], &samprowidx[0], G, &gram_size);
	dscal(&ngram, &alp, G, &incx);
		/*
		for(int i = 0; i < ngram; ++i)
			std::cout << G[i] << ' ';
		std::cout << std::endl;
		*/
		for(int i = 0; i < gram_size; ++i){
			//TODO: need to sample without replacement
			//if(rank ==0)
			//	std::cout << "index = " << index[i] << std::endl;
			//dcopy(&len, X + index[i], &m, Xsamp + i, &gram_size);
			wsamp[i] = w[index[i]];
		}


		//std::cout << "Xsamp[0] = " << Xsamp[0] << " Xsamp[1] = " << Xsamp[1] << std::endl;
		// Compute (s*b) x (s*b) Gram matrix


		//dgemm(&transa, &transb, &gram_size, &gram_size, &len, &alp, Xsamp, &gram_size, Xsamp, &gram_size, &zero, G, &gram_size);
		//std::cout << "dot product" << (1./n)*ddot(&len, Xsamp + 0, &gram_size, Xsamp + 1, &gram_size)+lambda << std::endl;



		// Compute y and alpha components of residual based on sampled rows.
		//std::cout << "Calling DGEMV" << std::endl;

		//std::cout << "len: " << len << " sampvals.length " << sampvals.size() << " sampcolidx.length " << sampcolidx.size() << " samprowidx.length " << samprowidx.size() << std::endl;
		mkl_dcsrmv(&transb, &len, &gram_size, &alp, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], alpha, &zero, G+(s*b*s*b));
		mkl_dcsrmv(&transb, &len, &gram_size, &alp, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], &y[0], &zero, G+(s*b*(s*b+1)));
		/*
		for(int i = 0; i < s*b; ++i)
			std::cout << *(G + (s*b*(s*b+1)) + i) << ' ';
		std::cout << std::endl;
		*/
		/*
		dgemv(&transa, &gram_size, &len, &alp, Xsamp, &gram_size, alpha, &incx, &zero, G + (s*b*s*b), &incx);
		dgemv(&transa, &gram_size, &len, &alp, Xsamp, &gram_size, y, &incx, &zero, G + (s*b*(s*b+1)), &incx);
		*/
		gramstp = MPI_Wtime();
		gramagg += gramstp - gramst;

		// Reduce and Broadcast: Sum partial Gram and partial residual components.
		//std::cout << "Calling ALLREDUCE" << std::endl;
		commst = MPI_Wtime();
		MPI_Allreduce(G,recvG,s*b*(s*b+2), MPI_DOUBLE, MPI_SUM, comm);
		commstp = MPI_Wtime();
		commagg += commstp - commst;
		//std::cout << "Adding lambda to diagonal" << std::endl;
		innerst = MPI_Wtime();
		for(int i =0; i < s*b; ++i)
				recvG[i + i*s*b] += lambda;
		/*
		if(rank == 0){
			for(int i = 0; i < s*b; ++i){
					for(int j = 0; j < s*b; ++j)
						std::cout << recvG[j*s*b + i] << " ";
					std::cout << std::endl;
				}
				std::cout << std::endl;
		}
		*/
		//MPI_Barrier(comm);

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

		//std::cout << "length 99" << std::endl;
		daxpy(&gram_size, &lambda, wsamp, &incx, recvG + s*b*s*b, &incx);
		daxpy(&gram_size, &neg, recvG + s*b*s*b, &incx, recvG + s*b*(s*b+1), &incx);

		dcopy(&gram_size, recvG + s*b*(s*b+1), &incx, del_w, &incx);
		/*
		std::cout << "residual on rank " << rank << " iter " << iter << std::endl;
		for(int i = 0; i < s*b; ++i){
				std::cout << del_w[i] << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;
		*/


		//compute solution to first (b) x (b) subproblem

		/*
		std::cout << "recvG[0] = " << recvG[0] << std::endl;
		std::cout << "before del_w = ";
		for(int j = 0; j < s*b; ++j){
			std::cout << del_w[j] << " ";
		}
		std::cout << std::endl;
		*/

		dpotrf(&uplo, &b, recvG, &gram_size, &info);
		assert(0==info);

		dpotrs(&uplo, &b, &nrhs, recvG, &gram_size, del_w, &b, &info);
		assert(0==info);
		for(int i = 0; i < b; ++i)
			w[index[i]] = w[index[i]] + del_w[i];
		iter++;
		innerstp = MPI_Wtime();
		inneragg += innerstp - innerst;

		if(iter == maxit){
			//gram_size = i*b;
			//dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_w, &incx, &one, alpha, &incx);
			free(alpha); free(G); free(recvG);
			free(index); free(del_w); free(wsamp); free(sampres);
			samprowidx.clear(); sampcolidx.clear(); sampvals.clear(); //sampres.clear();
			MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			if(rank == 0){
				std::cout << "Outer loop computation time: " << grammax*1000 << " ms" << std::endl;
				std::cout << "Inner loop computation time: " << innermax*1000 << " ms" << std::endl;
				std::cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << std::endl;
			}
			/*
			for(int i = 0; i < m; ++i)
				std::cout << w[i] << " ";
			std::cout << std::endl;
			*/
			return;
		}
		//std::cout << "Iter count: " << iter << std::endl;

		for(int i = 1; i < s; ++i){

			// Compute residual based on previous subproblem solution
			innerst = MPI_Wtime();
			lGcols = i*b;
			dgemv(&transa, &b, &lGcols, &neg, recvG + i*b, &gram_size, del_w, &incx, &one, del_w + i*b, &incx);

			// Correct residual if any sampled row in current block appeared in any previous blocks
			for(int j = 0; j < i*b; ++j){
				for(int k = 0; k < b; ++k){
					if(index[j] == index[i*b + k])
						del_w[i*b + k] -= lambda*del_w[j];
				}
			}

			// Compute solution to next (b) x (b) subproblem
			//std::cout << "recvG[" << i << "] = " << recvG[lGcols + s*lGcols] << std::endl;
			dpotrf(&uplo, &b, recvG + lGcols + s*b*lGcols, &gram_size, &info);
			assert(0==info);

			dpotrs(&uplo, &b, &nrhs, recvG + lGcols + s*b*lGcols, &gram_size, del_w + lGcols, &b, &info);
			assert(0==info);

			for(int j = 0; j < b; ++j)
				w[index[i*b + j]] = w[index[i*b + j]] + del_w[i*b + j];
			iter++;
			inneragg += MPI_Wtime() - innerst;
			if(iter == maxit){
				//gram_size = i*b;
				//dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_w, &incx, &one, alpha, &incx);
				free(alpha); free(G); free(recvG);
				free(index); free(del_w); free(wsamp); free(sampres);
				samprowidx.clear(); sampcolidx.clear(); sampvals.clear(); //sampres.clear();
				MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				if(rank == 0){
					std::cout << "Outer loop computation time: " << grammax*1000 << " ms" << std::endl;
					std::cout << "Inner loop computation time: " << innermax*1000 << " ms" << std::endl;
					std::cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << std::endl;
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

		//std::cout << "length 100" << std::endl;
		for(int i = 0; i < gram_size; ++i)
			sampres[index[i]] = recvG[gram_size*(gram_size+1) + i];

		ctol = dnrm2(&n, sampres, &incx);
		if(ctol <= tol){
				convcnt++;
				if(convthresh == convcnt){
					free(alpha); free(G); free(recvG);
					free(index); free(del_w); free(wsamp); free(sampres);
					samprowidx.clear(); sampcolidx.clear(); sampvals.clear(); //sampres.clear();
					MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					if(rank == 0){
						std::cout << "CA-BCD converged with estimated residual: " << std::scientific << ctol << std::setprecision(4) << std::fixed << " At outer iteration: " << iter/s << std::endl;
						std::cout << "Outer loop computation time: " << grammax*1000 << " ms" << std::endl;
						std::cout << "Inner loop computation time: " << innermax*1000 << " ms" << std::endl;
						std::cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << std::endl;
					}
					return;
				}
				//else{
				//	convcnt = 0; conviter = iter;
				//}
		}

		/*
		std::cout << "after del_w = ";
		for(int j = 0; j < s*b; ++j){
			std::cout << del_w[j] << " ";
		}
		std::cout << std::endl;
		*/

		// Update w
		//std::cout << "w = ";
		//	std::cout << w[index[j]] << " ";
		//std::cout << std::endl;


		// Update local alpha
		gramst = MPI_Wtime();
		mkl_dcsrmv(&transa, &len, &gram_size, &one, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], del_w, &one, alpha);
		//dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_w, &incx, &one, alpha, &incx);
		gramagg += MPI_Wtime() - gramst;



		sampcolidx.clear(); sampvals.clear(); //sampres.clear();
		samprowidx[0] = 1;
		memset(G, 0, sizeof(double)*gram_size*(gram_size+2));
		memset(recvG, 0, sizeof(double)*gram_size*(gram_size+2));
		memset(del_w, 0, sizeof(double)*gram_size);
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
	parse_lines_to_csr(lines, rowidx, colidx, vals, y, dual_method);
		/*
		for(int i = 0; i < y.size(); ++i)
			std::cout << y[i] << ' ';
			std::cout << std::endl;
		*/
	//compute scatter offsets
/*
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
	*/
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
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0)
		std::cout << "Starting warm-up call" << std::endl;
	cabcd(rowidx, colidx, vals, m, n, y, y.size(), lambda, s, b, maxit, tol, seed, freq, w, comm);
	if(rank == 0){
		std::cout << "w = ";
		for(int i = 0; i < n; ++i)
			std::cout << std::setprecision(4) << std::fixed << w[i] << " ";
		std::cout << std::endl;
	}

	s = 1;
	for(int k = 0; k < 4; ++k){
		if(b > n)
			continue;
		for(int j = 0; j < 7; ++j){
			if(rank == 0){
				std::cout << std::endl << std::endl;
				std::cout << "s = " << s << ", " << "b = " << b << std::endl;
			}
			//cabcd(localX, n, m, localy, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
			cabcd(rowidx, colidx, vals, m, n, y, y.size(), lambda, s, b, maxit, tol, seed, freq, w, comm);
			algst = MPI_Wtime();
			for(int i = 0; i < niter; ++i){
				//cabcd(localX, n, m, localy, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
				cabcd(rowidx, colidx, vals, m, n, y, y.size(), lambda, s, b, maxit, tol, seed, freq, w, comm);
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
			double algmaxt = 1000*(algstp - algst)/niter;
			//double algmax;
			//MPI_Reduce(&algmaxt, &algmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			if(rank == 0)
				std::cout << std::endl << "Total CA-BCD time: " << algmaxt << " ms" << std::endl;
			s *= 2;
		}
		s = 1;
		b *= 2;
	}
	/*
	if(rank == 0){
		std::cout << "w = ";
		for(int i = 0; i < n; ++i)
			std::cout << std::setprecision(4) << std::fixed << w[i] << " ";
		std::cout << std::endl;
	}
	*/
	/*
	free(localX); free(localy);
	free(cnts); free(displs);
	free(cnts2); free(displs2);
	*/
	MPI_Finalize();
	return 0;
}
