#include <iostream>
#include <mkl.h>
#include "mpi.h"
#include <assert.h>
#include <stdlib.h>
#include <cstring>
#include <string.h>
#include <iomanip>
#include <math.h>
#include <algorithm>

#include "calasso.h"
#include "util.h"

void calasso(			std::vector<int> &rowidx,
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
						double *v,
						MPI_Comm comm,
						double theta_start)
{
	int npes, rank;
	MPI_Comm_size(comm, &npes);
	MPI_Comm_rank(comm, &rank);

	int thetaidx = -1;
	double *alpha, *res,  *obj_err, *sol_err;
	double *tw, *talpha, *theta, *c_scal;
	double *del_w;
	double ctol, stepsize, c;
	double *G, *recvG, *Xsamp, *wsamp, *sampres, *sampres_sum;
	int incx = 1;
	int rescnt = 0;
	int *index;
	int blk_start, matvec_offset;
	int blk_end, blk_len;
	int nblks = (n % b == 0) ? (n/b) : (n/b + 1);
	int gram_size = s*b;
	int ngram = s*b*s*b;

	assert(0==Malloc_aligned(double, alpha, n, ALIGN));
	assert(0==Malloc_aligned(double, talpha, len, ALIGN));
	assert(0==Malloc_aligned(double, tw, len, ALIGN));
	assert(0==Malloc_aligned(double, sampres, len, ALIGN));

	assert(0==Malloc_aligned(double, G, gram_size*(gram_size + 2), ALIGN));
	assert(0==Malloc_aligned(double, recvG, s*b*(s*b + 2), ALIGN));
	
	assert(0==Malloc_aligned(double, del_w, s*b, ALIGN));
	assert(0==Malloc_aligned(double, res, s*b, ALIGN));
	assert(0==Malloc_aligned(double, wsamp, s*b, ALIGN));
	
	assert(0==Malloc_aligned(int, index, s, ALIGN));
	assert(0==Malloc_aligned(double, theta, s+1, ALIGN));
	assert(0==Malloc_aligned(double, c_scal, s, ALIGN));
	assert(0==Malloc_aligned(double, sampres_sum, n, ALIGN));
	

	theta[0] = theta_start;

	//if(rank == 0)
	//std::cout << "Initialized alpha and w to 0" << std::endl;
	memset(alpha, 0, sizeof(double)*n);
	memset(w, 0, sizeof(double)*n);
	memset(wsamp, 0, sizeof(double)*s*b);
	memset(talpha, 0, sizeof(double)*len);
	memset(sampres, 0, sizeof(double)*len);
	memset(tw, 0, sizeof(double)*len);

	char transa = 'N', transb = 'T', uplo = 'U';
	double alp = 1./m;
	double one = 1., zero = 0., neg = -1.;
	double neg_lambda = -lambda;
	double neg_alp = -alp;
	double resdot = 0., recvresdot = 0.;
	double old_obj = 0., objdiff = -1.;
	int info, nrhs = 1;
	int lGcols = b;
	char matdesc[6];
	matdesc[0] = 'G'; matdesc[3] = 'F';
	srand(seed);
	
	daxpy(&len, &neg, &y[0], &incx, tw, &incx);

	double commst = 0., commstp = 0., commagg = 0.;
	double gramst= 0., gramstp= 0., gramagg = 0.;
	double innerst= 0., innerstp= 0., inneragg = 0.;
	int iter = 0;
	int offset = 0;

	int convcnt = 0;
	int conviter = 0;
	int convthresh = (n%b == 0) ? n/b : n/b + 1;

	int cursamp, count;
	std::vector<int> padded_blks(nblks, 0);
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
		//if(rank == 0)
		//std::cout << "Choosing the following blocks: ";
		cursamp = 0;
		count = 0;
		while(cursamp < s){
			count = (rand() % (int)(nblks));
			while(v[count] == 0.){
				count = (rand() % (int)(nblks));
			}
			index[cursamp] = count;
			cursamp++;
			//if(rank == 0)
			//std::cout << count << ':' << v[count] << ' ';
		}
		//std::cout << std::endl;
		//if(rank == 0)
		//std::cout << std::endl << " Finished selecting blocks" << std::endl;

		/*
		for(int i =  rowidx[0]; i < rowidx[1]; ++i)
			std::cout << colidx[i-1] << ":" << vals[i-1] << ' ';
		std::cout << std::endl;
		*/

		for(int i = 1; i < rowidx.size(); ++i){
			samprowidx[i] = samprowidx[i-1];
			//std::cout << samprowidx.size()<< std::endl;
			for(int k = 0; k < s; ++k){
				blk_start = index[k]*b;
				blk_end = (index[k]+1)*b;
				blk_end = (blk_end > n) ? (n) : (blk_end);
				for(int j = rowidx[i-1]-1; j < rowidx[i]-1; ++j){
					cidx = colidx[j];
					tval = vals[j];
					if(cidx >= blk_start+1 && cidx < blk_end+1){
						//std::cout << "currently i = " << i << std::endl;
						sampcolidx.push_back(cidx - blk_start+ k*b );
						sampvals.push_back(tval);
						//std::cout << "Incrementing samprowidx[" << cidx - blk_start + k*b << "] to " << samprowidx[i] + 1 << std::endl;
						samprowidx[i]++;
					}
				}
			
				/*If current block is less than blocksize, then pad with zeros until # columns = blocksize
				 *Save computation and storage by padding just 1 row (hence the vector padded_blks.)
				 *If we padded the current block, then don't bother doing it again.
				 */
				/*
				if((blk_end - blk_start) != b && padded_blks[k] == 0){
					for(int padcnt = 0; padcnt < b - (blk_end - blk_start); ++padcnt){
						sampcolidx.push_back(blk_end - blk_start + padcnt + 1 + k*b);
						sampvals.push_back(0.);
						samprowidx[i]++;
					}
					padded_blks[k] = 1;
					//std::cout << " Just padded block: " << k << std::endl;
				}
				*/
			}
		}
		/*
		for(int i =  samprowidx[0]; i < samprowidx[1]; ++i)
			std::cout << sampcolidx[i-1] << ":" << sampvals[i-1] << ' ';
		std::cout << std::endl;
		*/

		//std::cout << "Computing Gram matrix" << std::endl;
		if(s > -1){
			/** Need to compute off-diagonal blocks of Gram matrix (This is unoptimized... currently computing the full Gram matrix.) **/
			mkl_dcsrmultd(&transb, &len, &gram_size, &gram_size, &sampvals[0], &sampcolidx[0], &samprowidx[0],  &sampvals[0], &sampcolidx[0], &samprowidx[0], G, &gram_size);

			mkl_dcsrmv(&transb, &len, &gram_size, &one, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], tw, &zero, G+(s*b*s*b));
			mkl_dcsrmv(&transb, &len, &gram_size, &one, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], talpha, &zero, G+(s*b*(s*b+1)));
			gramstp = MPI_Wtime();
			gramagg += gramstp - gramst;
			commst = MPI_Wtime();
			MPI_Allreduce(G, recvG, ngram + 2*s*b, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			commstp = MPI_Wtime();
			commagg += commstp - commst;
			//std::cout << std::setprecision(20) << std::fixed;
			
			/*
			if(rank == 0){
				std::cout << "recvG = " << std::endl;
				for(int i =0; i < 2*b; ++i){
					for(int j = 0; j < 2*b; ++j){
						std::cout << recvG[i*gram_size + j] << ' ';
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;

			}
			*/

			innerst = MPI_Wtime();
			for(int i = 0; i < s; ++i){
				c = theta[i]*theta[i];
				//if(rank == 0)
				//	std::cout << "theta[" << i << "] = " << theta[i] << std::endl;
				daxpy(&b, &c, recvG+ngram+(s*b)+(i*b), &incx, recvG+ngram+(i*b), &incx);
				theta[i+1] = (sqrt((c*c) + (4*c)) - c)/2;
				c_scal[i] = (1. - (double)nblks*theta[i])/theta[i]/theta[i];
			}
			/*
			std::cout << "res = " << std::endl;
			for(int j = 0; j < gram_size; ++j){
				std::cout << recvG[ngram + j] << ' ';
			}
			std::cout << std::endl;
			*/


			//daxpy(&gram_size, &one, recvG+ngram+s*b, &incx, recvG+ngram, &incx);
			dcopy(&gram_size, recvG + ngram, &incx, res, &incx);
			/*
			if(rank == 0){
				std::cout << "Residual = " << std::endl;
				for(int i = 0; i < gram_size; ++i)
					std::cout << res[i] << ' ';
				std::cout << std::endl;
			}
			*/
		}
		else{
			/** Since the Lipschitz constants are pre-coumpted we only need to compute the residual. **/
			mkl_dcsrmultd(&transb, &len, &gram_size, &gram_size, &sampvals[0], &sampcolidx[0], &samprowidx[0],  &sampvals[0], &sampcolidx[0], &samprowidx[0], G, &gram_size);
			mkl_dcsrmv(&transb, &len, &gram_size, &one, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], tw, &zero, G + ngram);
			mkl_dcsrmv(&transb, &len, &gram_size, &one, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], talpha, &zero, G+ ngram + s*b);
			//std::cout << std::setprecision(20) << std::fixed;
			gramstp = MPI_Wtime();
			gramagg += gramstp - gramst;
			
			commst = MPI_Wtime();
			MPI_Allreduce(G, recvG, ngram + 2*s*b, MPI_DOUBLE, MPI_SUM, comm);
			commstp = MPI_Wtime();
			commagg += commstp - commst;
			
			innerst = MPI_Wtime();
			c = theta[0]*theta[0];
			daxpy(&gram_size, &c, recvG + ngram +s*b, &incx, recvG + ngram, &incx);
			theta[s] = (sqrt(c*c + 4*c) - c)/2;
			c_scal[0] = (1. - (double)nblks*theta[0])/theta[0]/theta[0];


			
			//daxpy(&gram_size, &one, G+s*b, &incx, G, &incx);
			/*
			if(rank == 0){
				std::cout << "Residual = " << std::endl;
				for(int i = 0; i < gram_size; ++i)
					std::cout << recvG[i] << ' ';
				std::cout << std::endl;
			}
			*/

			dcopy(&gram_size, recvG + ngram, &incx, res, &incx);
		}
		/*
		 * Inner s-step loop
		 * Perfomed redundantly on all processors
		*/
		for(int i = 0; i < s; ++i){
			
			blk_start = index[i]*b;
			blk_end = (index[i]+1)*b;
			blk_end = (blk_end > n) ? (n) : blk_end;
			
			for(int j = blk_start, k = 0; j < blk_end; ++j, ++k){
				wsamp[i*b + k] = w[j];
			//	if(rank == 0)
			//		std::cout << wsamp[i*b + k] << ' ';
			}
			//std::cout << std::endl;
		}

		dcopy(&gram_size, wsamp, &incx, del_w, &incx); 
		stepsize = (((-1./((double)nblks))/theta[0])/v[index[0]]);
		
		/*
		if(rank == 0){
			std::cout << std::setprecision(20) << "Stepsize: " << stepsize << std::endl;
			std::cout << "theta[0]: " << theta[0] << std::endl;
			std::cout << "v[index[0]]: " << v[index[0]] << std::endl;
			
			std::cout << "Lambda: " << lambda << std::endl;
		}
		*/
		/** Compute the gradient update **/
		daxpy(&b, &stepsize, res, &incx, del_w, &incx);
		
		/*
		if(rank == 0){
			std::cout << "Gradient: ";
			for(int i = 0; i < b; ++i){
				std::cout << del_w[i] << ' ';
			}
			std::cout << std::endl;
		}
		*/

		/** Compute the soft-thresholding solution **/
		
		//if(rank == 0)
			//std::cout << "Solution update: " << std::endl;
		for(int i = 0; i < b; ++i){
			del_w[i] = sign(del_w[i])*max(fabs(del_w[i]) + lambda*stepsize, 0.) - wsamp[i];
			//if(rank == 0)
			//	std::cout << del_w[i] << ' ';
		}
		//std::cout<< std::endl;
		//if(rank == 0)
		//	std::cout << std::endl;

		//combine residual updates into one vector

		/*
		if(iter == maxit){
			//gram_size = i*b;
			//dgemv(&transb, &gram_size, &len, &one, Xsamp, &gram_size, del_w, &incx, &one, alpha, &incx);
			free(alpha); free(G); free(recvG);
			free(index); free(del_w); free(wsamp); 
			free(sampres);
			samprowidx.clear(); sampcolidx.clear(); sampvals.clear(); //sampres.clear();
			MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			if(rank == 0){
				std::cout << "Outer loop computation time: " << grammax*1000 << " ms" << std::endl;
				std::cout << "Inner loop computation time: " << innermax*1000 << " ms" << std::endl;
				std::cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << std::endl;
			}
			return;
		}
		*/
		for(int i = 1; i < s; ++i){

			// Compute residual based on prev
			stepsize = (((-1./((double)nblks))/theta[i])/v[index[i]]);
			//if(rank == 0)
			//	std::cout << "inner loop stepsize = " << stepsize << std::endl;
			
			dcopy(&b, res + i*b, &incx, del_w + i*b, &incx);
			
			/*
			if(rank == 0){
				std::cout << "s-step iteration: " << i << " res = ";
				for(int j = 0; j < b; ++j){
					std::cout << res[i*b + j] << ' ';
				}
				std::cout << std::endl;
			}
			*/
			for(int j = 0; j < i; ++j){	
				c = -c_scal[j]*theta[i]*theta[i] + 1.;
				dgemv(&transb, &b, &b, &c, recvG + s*b*b*i + j*b, &gram_size, del_w + j*b, &incx, &one, del_w + i*b, &incx);
			}
			/*
			if(rank == 0){
				std::cout << "s-step iteration: " << i << " gemv result = "; 
				for(int j = 0; j < b; ++j)
					std::cout << del_w[i*b+ j] << ' ';
				std::cout << std::endl;
			}
			*/
			
			dscal(&b, &stepsize, del_w + i*b, &incx);
			daxpy(&b, &one, wsamp + i*b, &incx, del_w + i*b, &incx);
			
			for(int j = 0; j < i; ++j){
				if(index[j] == index[i]){
					daxpy(&b, &one, del_w + j*b, &incx, del_w + i*b, &incx);
				}
			}
			for(int j = 0; j < b; ++j){
				del_w[j + i*b] = sign(del_w[j + i*b])*max(fabs(del_w[j + i*b]) + lambda*stepsize, 0.) - wsamp[j + i*b];
			}
			for(int j = 0; j < i; ++j){
				if(index[j] == index[i]){
					daxpy(&b, &neg, del_w + j*b, &incx, del_w + i*b, &incx);
				}
			}
			/*
			if(rank == 0){
				std::cout << "s-step iteration: " << i << " del_w = "; 
				for(int j = 0; j < b; ++j)
					std::cout << del_w[i*b+ j] << ' ';
				std::cout << std::endl;
			}
			*/
			/*if(iter == maxit){
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
				return;
			}
			*/
		}


		// TODO: Update alpha, talpha, w, and tw (assumes that blocks are contiguous (when updating w)
		mkl_dcsrmv(&transa, &len, &gram_size, &one, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], del_w, &one, tw);
		for(int i = 0; i < s; ++i){
			blk_start = index[i]*b;
			blk_end = (index[i]+1)*b;
			blk_end = (blk_end > n) ? n : blk_end;
			blk_len = blk_end - blk_start;
			
			daxpy(&blk_len, &one, del_w + i*b, &incx, w + blk_start, &incx);
			dscal(&b, &c_scal[i], del_w + i*b, &incx);
			daxpy(&blk_len, &neg, del_w + i*b, &incx, alpha + blk_start, &incx);
			iter++;

			if(iter/freq == rescnt){
				thetaidx = i;
			}
		}
		
		mkl_dcsrmv(&transa, &len, &gram_size, &neg, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], del_w, &one, talpha);
		/*
		if(rank == 0){
			std::cout << "Solution w = ";
			for(int i = 0; i < n; ++i){
				std::cout << w[i] << ' ';
			}
			std::cout << std::endl;
			std::cout << "Solution alpha = ";
			for(int i = 0; i < n; ++i){
				std::cout << alpha[i] << ' ';
			}
			std::cout << std::endl;
			
			std::cout << "Solution tw = ";
			for(int i = 0; i < 10; ++i){
				std::cout << tw[i] << ' ';
			}
			std::cout << std::endl;
			std::cout << "Solution talpha = ";
			for(int i = 0; i < 10; ++i){
				std::cout << talpha[i] << ' ';
			}
			std::cout << "Iter = " << iter << std::endl;
		}
		*/
		innerstp = MPI_Wtime();
		inneragg += innerstp - innerst;
		if(iter/freq == rescnt){
			c = theta[thetaidx]*theta[thetaidx];
			dcopy(&len, tw, &incx, sampres, &incx);
			daxpy(&len, &c, talpha, &incx, sampres, &incx);
			resdot = ddot(&len, sampres, &incx, sampres, &incx);
			
			/*
			if(rank == 0){
				std::cout << "sampres = ";
				for(int i = 0; i < len; ++i)
					std::cout << sampres[i] << ' ';
				std::cout << std::endl;
				std::cout << std::setprecision(20) << "Iteration: " << iter << " c = " << c  << " partial F(x) = " <<  resdot << std::endl;
			}
			*/
			recvresdot = 0.;
			MPI_Allreduce(&resdot, &recvresdot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			recvresdot /= 2.;
			for(int i = 0; i < n; ++i){
				recvresdot += lambda*fabs(c*alpha[i] + w[i]); 
			}
			
			if(rank == 0){
				std::cout <<  recvresdot << std::endl;
			}
			objdiff = (objdiff == -1.) ? (recvresdot) : (fabs(old_obj - recvresdot));
			old_obj = recvresdot;
			rescnt++;
		}
		
		if(iter >= maxit || objdiff <= tol){
			c = theta[thetaidx]*theta[thetaidx];
			daxpy(&n, &c, alpha, &incx, w, &incx);
			if(rank == 0)
				std::cout << "Objdiff = " << objdiff << std::endl;
			/*if(rank == 0){
				std::cout << "FINAL SOLUTION: ";
				for(int i = 0; i < n; ++i)
					std::cout << w[i] << ' ';
				std::cout << std::endl;
			}
			*/
			free(alpha); free(G); free(recvG);
			free(index); free(del_w); free(wsamp); free(sampres);
			free(tw); free(talpha); free(res);
			free(theta); free(c_scal);
			samprowidx.clear(); sampcolidx.clear(); sampvals.clear(); //sampres.clear();
			MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			if(rank == 0){
				std::cout << std::setprecision(3) << "Outer loop computation time: " << (grammax*1000)/(double)npes << " ms" << std::endl;
				std::cout << std::setprecision(3) << "Inner loop computation time: " << (innermax*1000)/(double)npes << " ms" << std::endl;
				std::cout << std::setprecision(3) << "MPI_Allreduce time: " << (commmax*1000)/(double)npes << " ms" << std::endl;
			}
			return;
		}
		theta[0] = theta[s];
		
		/** Left here from Ridge algorithm
		gramst = MPI_Wtime();
		mkl_dcsrmv(&transa, &len, &gram_size, &one, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], del_w, &one, alpha);
		if(iter/freq == rescnt){
			mkl_dcsrmv(&transb, &len, &n, &neg_alp, matdesc, &vals[0], &colidx[0], &rowidx[0], &rowidx[1], alpha, &zero, sampres);
			mkl_dcsrmv(&transb, &len, &n, &alp, matdesc, &vals[0], &colidx[0], &rowidx[0], &rowidx[1], &y[0], &one, sampres);

			MPI_Allreduce(sampres, sampres_sum, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			daxpy(&n, &neg_lambda, w, &incx, sampres_sum, &incx);
			resnrm = dnrm2(&n, sampres_sum, &incx);
			rescnt++;
			if(resnrm <= tol){
					free(alpha); free(G); free(recvG);
					free(index); free(del_w); free(wsamp); free(sampres);
					samprowidx.clear(); sampcolidx.clear(); sampvals.clear(); //sampres.clear();
					MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					if(rank == 0){
						std::cout << "CA-BCD converged with residual: " << std::scientific << resnrm << std::setprecision(4) << std::fixed << " At outer iteration: " << iter/s << std::endl;
						std::cout << "Outer loop computation time: " << grammax*1000 << " ms" << std::endl;
						std::cout << "Inner loop computation time: " << innermax*1000 << " ms" << std::endl;
						std::cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << std::endl;
					}
					return;
			
			}
		}
		gramagg += MPI_Wtime() - gramst;

		**/

		sampcolidx.clear(); sampvals.clear(); //sampres.clear();
		samprowidx[0] = 1;
		memset(G, 0, sizeof(double)*gram_size*(gram_size+2));
		memset(recvG, 0, sizeof(double)*gram_size*(gram_size+2));
		memset(del_w, 0, sizeof(double)*gram_size);
		memset(wsamp, 0, sizeof(double)*gram_size);
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
			std::cout << argv[0] << " [filename] [rows] [cols] [lambda] [maxit] [tol] [seed] [freq] [block size] [loop block size] [number of benchmark iterations] [frequency of residual computation]" << std::endl;
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
	int flag = atoi(argv[12]);

	int sorig = s;
	std::string lines = libsvmread(fname, m, n);

	std::vector<int> rowidx, colidx;
	std::vector<double> y, vals;

	int dual_method = 0;
	parse_lines_to_csr(lines, rowidx, colidx, vals, y, dual_method, m, n);
	//std::cout << y.size() << std::endl;
	//	for(int i = 0; i < y.size(); ++i)
	//		std::cout << y[i] << " ";
	//	std::cout << std::endl;
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
	double *w, *v, *recv;
	double *nrms;
	b= 1;
	int nblks = (n % b == 0) ? (n/b) : (n/b + 1);
	//std::cout << "nblks = " << nblks << std::endl;
	//std::cout << "nrows = " << rowidx.size()-1 << std::endl;
	assert(0==Malloc_aligned(double, w, n, ALIGN));
	assert(0==Malloc_aligned(double, v, n, ALIGN));
	assert(0==Malloc_aligned(double, recv, n, ALIGN));
	assert(0==Malloc_aligned(double, nrms, nblks, ALIGN));
	

	/*Compute the Lipschitz constants for each block (tau = 1, for now).*/
	int blk_start, blk_end;
	for(int i = 0; i < nblks; ++i){
		blk_start = (i*b);
		blk_end = (i+1)*b;
		if(blk_end > n)
			blk_end = n;
		/*Compute ||.||_2^2 of each column in this blk.*/
		for(int j = blk_start; j < blk_end; ++j){
			v[j] = 0.;
			for(int k = 0; k < colidx.size(); ++k){
				if(colidx[k] -1 == j)
					v[j] += vals[k]*vals[k];
			}
		}
	}
	MPI_Allreduce(v, recv, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	for(int i = 0; i < nblks; ++i){
		blk_start = (i*b);
		blk_end = (i+1)*b;
		
		if(blk_end > n)
			blk_end = n;

		nrms[i] = 0.;
		for(int j =blk_start; j < blk_end; ++j){
			nrms[i] += recv[j];
		}
	}

	/*
	if(rank == 0){
		std::cout << "Nrms: ";
		for(int i =0; i < nblks; ++i)
			std::cout << nrms[i] << ' ';
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
	}
	*/

	/*if(rank == 0){
		std::cout << "Lipschitz constants: ";
		for(int i = 0; i < nblks; ++i){
			std::cout << recv[i] << ' ';
		}
		std::cout << std::endl;
	}*/
	
	int s_lim = 9;
	std::cout << std::setprecision(3);
	
	
	double theta_start = (flag == 0) ? (1.) : (1./(double)nblks);
	for(int k = 0; k < 1; ++k){
		if(b > n)
			continue;
		for(int j = 0; j <s_lim; ++j){
			if(rank == 0){
				std::cout << std::endl << std::endl;
				std::cout << "s = " << s << ", " << "b = " << b << std::endl;
			}
			//cabcd(rowidx, colidx, vals, m, n, y, y.size(), lambda, s, b, maxit, tol, seed, freq, w, comm);
			MPI_Barrier(MPI_COMM_WORLD);
			algst = MPI_Wtime();
			for(int i = 0; i < niter; ++i){
				//cabcd(localX, n, m, localy, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
				calasso(rowidx, colidx, vals, m, n, y, y.size(), lambda, s, b, maxit, tol, seed, freq, w, nrms,comm, theta_start);
			}
			algstp = MPI_Wtime();
			double algmaxt = 1000*(algstp - algst)/niter;
			//double algmax;
			//MPI_Reduce(&algmaxt, &algmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			if(rank == 0){
				std::cout << std::endl << std::setprecision(3) << "Total CA-LASSO time: " << algmaxt << " ms" << std::endl;
			}
			MPI_Barrier(MPI_COMM_WORLD);
			s *= 2;
		}
		s = sorig;
		b *= 8;
		//s_lim -=3;
		nblks = (n % b == 0) ? (n/b) : (n/b + 1);
		MPI_Barrier(MPI_COMM_WORLD);


		/*Compute the Lipschitz constants for each block (tau = 1, for now).*/
		for(int i = 0; i < nblks; ++i){
			blk_start = (i*b);
			blk_end = (i+1)*b;
			if(blk_end > n)
				blk_end = n;
			/*Compute ||.||_2^2 of each column in this blk.*/
			for(int j = blk_start; j < blk_end; ++j){
				v[j] = 0.;
				for(int k = 0; k < colidx.size(); ++k){
					if(colidx[k] -1 == j)
						v[j] += vals[k]*vals[k];
				}
			}
		}
		MPI_Allreduce(v, recv, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
		for(int i = 0; i < nblks; ++i){
			blk_start = (i*b);
			blk_end = (i+1)*b;
			
			if(blk_end > n)
				blk_end = n;

			nrms[i] = 0.;
			for(int j =blk_start; j < blk_end; ++j){
				nrms[i] += recv[j];
			}
		}
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
