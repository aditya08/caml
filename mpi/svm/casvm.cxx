/************************************
Copyright (c) 2017 Aditya Devarakonda, all rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**************************************/

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

#include "casvm.h"
#include "util.h"

void casvm(				std::vector<int> &rowidx,
						std::vector<int> &colidx,
						std::vector<double> &vals,	//input args
						int m,
						int n,
						std::vector<double> &y,
						int len,
						double lambda,
						int s,
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


	double *del_alp, *G, *recvG, *res, *wsamp, *tmpalp, *tmpy;
	double *alpha, *grad, *proj_grad, *theta, *tildew, *tildewsum;
	int *index;
	assert(0 == Malloc_aligned(double, del_alp, s, ALIGN));
	assert(0 == Malloc_aligned(double, G, s*(s + 2), ALIGN));
	assert(0 == Malloc_aligned(double, recvG, s*(s + 2), ALIGN));
	assert(0 == Malloc_aligned(double, res, s, ALIGN));
	assert(0 == Malloc_aligned(double, tmpalp, s, ALIGN));
	assert(0 == Malloc_aligned(double, tmpy, s, ALIGN));
	assert(0 == Malloc_aligned(double, alpha, n, ALIGN));
	assert(0 == Malloc_aligned(double, tildew, n, ALIGN));
	assert(0 == Malloc_aligned(double, tildewsum, n, ALIGN));
	assert(0 == Malloc_aligned(double, grad, s, ALIGN));
	assert(0 == Malloc_aligned(double, proj_grad, s, ALIGN));
	assert(0 == Malloc_aligned(double, theta, s, ALIGN));
	assert(0 == Malloc_aligned(int, index, s, ALIGN));

	if(rank == 0)
		std::cout << "matrix dimensions: " << m << " x " << n << std::endl;
	//std::cout << "nrows = " << len << std::endl;
	//std::cout << "rowidx = " << rowidx.size() - 1 << std::endl;


	std::fill(w, w+len, 0.);
	std::fill(alpha, alpha + n, 0.);
	std::fill(G, G + (s*s + s), 0.);
	std::fill(recvG, recvG + (s*s + s), 0.);
	
	/*
	memset(w, 0, sizeof(double)*len);
	memset(alpha, 0, sizeof(double)*n);
	memset(G, 0, sizeof(double)*s*(s + 1));
	memset(recvG, 0, sizeof(double)*s*(s + 1));
	*/

	char transa = 'N', transb = 'T', uplo = 'U';
	char matdesc[6];
	double one = 1., zero = 0.;
	int incx = 1, rescnt = 0;
	matdesc[0] = 'G'; matdesc[3] = 'F';
	srand48(seed);
	
	double commst, commstp, commmax, commagg = 0.;
	double gramst, gramstp, grammax, gramagg = 0.;
	double resst, resstp, resmax, resagg = 0.;
	double innerst, innerstp, innermax, inneragg = 0.;
	double matcpyst, matcpystp, matcpymax, matcpyagg = 0.;
	
	int iter = 0;

	int cursamp, count;
	int cidx = 0;
	double tval = 0., gradcorr = 0.;

	std::vector<int> samprowidx(rowidx.size(), 1);
	std::vector<int> sampcolidx;
	std::vector<double> sampvals;
	
	while(1){
		
		/*only need dot-product, so don't copy anything.*/
		if(s == 1){
			matcpyst = MPI_Wtime();
			cursamp = 0;
			count = 0;
			while(cursamp < 1){
				if(((n-count)*drand48()) >= (1 - cursamp))
					++count;
				else{
					index[cursamp] = count;
					//index[cursamp] = iter%n;
					++count; ++cursamp;
				}
			}
			//if(rank == 0)
			//	std::cout << "Chosen index: " << index[0] << std::endl;
			tmpalp[0] = alpha[index[0]];
			tmpy[0] = y[index[0]];
			for(int i = 1; i < rowidx.size(); ++i){
				samprowidx[i] = samprowidx[i-1];
				for(int j = rowidx[i-1]-1; j < rowidx[i]-1; ++j){
					cidx = colidx[j];
					tval = vals[j];
					if(cidx == index[0]+1){
						/*Compute the dot-products right now*/
						gramst = MPI_Wtime();
						G[0] += tval*tval;
						G[1] += tval*w[i-1];
						gramstp = MPI_Wtime();
						gramagg += gramstp - gramst;
						/*Store the column of A for the axpy to update w later.*/
						sampcolidx.push_back(1);
						sampvals.push_back(tval);
						samprowidx[i]++;
					}
				}
			}

			//std::cout << std::endl;
			/*Count only the time for the sampling + matrix copy.*/
			matcpystp = MPI_Wtime();
			matcpyagg += (matcpystp - matcpyst);
			
			commst = MPI_Wtime();
			MPI_Allreduce(G, recvG, 2, MPI_DOUBLE, MPI_SUM, comm);
			commstp = MPI_Wtime();
			commagg += commstp - commst;
		}

		/* s-step needs a DGEMM + SpMV so copy into temporary CSR buffer*/
		else{
			matcpyst = MPI_Wtime();
			for(int i = 0; i < s; ++i){
				cursamp = 0;
				count = 0;
				while(cursamp < 1){
					if(((n-count)*drand48()) >= (1 - cursamp))
						++count;
					else{
						index[cursamp + i] = count;
						tmpy[i] = y[count];
						tmpalp[i] = alpha[count];
						//if(rank == 0)
							//std::cout << count << ' ';
						++count; ++cursamp;
					}
				}
			}
			for(int i = 1; i < rowidx.size(); ++i){
				samprowidx[i] = samprowidx[i-1];
				//std::cout << samprowidx.size()<< std::endl;
				for(int k = 0; k < s; ++k){
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
			matcpystp = MPI_Wtime();
			matcpyagg += matcpystp - matcpyst;
			
			gramst = MPI_Wtime();
			mkl_dcsrmultd(&transb, &len, &s, &s, &sampvals[0], &sampcolidx[0], &samprowidx[0],  &sampvals[0], &sampcolidx[0], &samprowidx[0], G, &s);
		mkl_dcsrmv(&transb, &len, &s, &one, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], w, &zero, G+s*s);
			gramstp = MPI_Wtime();
			gramagg += gramstp - gramst;

			commst = MPI_Wtime();
			MPI_Allreduce(G, recvG, s*s+s, MPI_DOUBLE, MPI_SUM, comm);
			commstp = MPI_Wtime();
			commagg += commstp - commst;

		}
		//if(rank == 0)
		//	std::cout << "eta = " << recvG[0]  << " res = " << recvG[1]<< std::endl;
	
		innerst = MPI_Wtime();
		/*Unroll the s-step iterations by 1.*/
		grad[0]= tmpy[0]*recvG[s*s] - 1;

		proj_grad[0] = fabs(fmin(fmax(tmpalp[0] - grad[0], 0.0), lambda) - tmpalp[0]);
		
		if(proj_grad[0] > 1e-14)
			theta[0] = fmin(fmax(tmpalp[0] - grad[0]/recvG[0], 0.0), lambda) - tmpalp[0];
		else
			theta[0] = 0.;

		
		//if(rank == 0)
		//	std::cout << "Gradient = " << grad[0]  << " projected_Gradient = " << proj_grad[0] << " theta = " << theta[0] << std::endl;


		/*Update coordinate of alpha*/
		alpha[index[0]] += theta[0];
		theta[0] *= tmpy[0];
		/*Update all of w, iff s = 1*/
		if(s == 1){
			for(int i = 1; i < samprowidx.size(); ++i){
				//std::cout << samprowidx.size()<< std::endl;
				for(int j = samprowidx[i-1]-1; j < samprowidx[i]-1; ++j){
					 w[i-1] += theta[0]*tmpy[0]*sampvals[j];
				}
			}
		}

		iter++;

		/*perform remaining s-step loop if required*/
		for(int i = 1; i < s; ++i){
			grad[i] = tmpy[i]*recvG[s*s + i] - 1;

			for(int j = 0; j < i; ++j){
				gradcorr += recvG[i*s + j]*theta[j];
			}

			grad[i] += gradcorr*tmpy[i];

			for(int j = 0; j < i; ++j){
				if(index[i] == index[j]){
					tmpalp[i] += theta[j];
				}
			}

			proj_grad[i] = fabs(fmin(fmax(tmpalp[i] - grad[i], 0.0), lambda) - tmpalp[i]);
			if(proj_grad[i] > 1e-14)
				theta[i] = fmin(fmax(tmpalp[i] - grad[i]/recvG[s*i + i], 0.0), lambda) - tmpalp[i];
			else
				theta[i] = 0.;
			alpha[index[i]] += theta[i];
			theta[i] *= tmpy[i];
			iter++;
		}
		innerstp = MPI_Wtime();
		inneragg += innerstp - innerst;
		if(s > 1){
			mkl_dcsrmv(&transa, &len, &s, &one, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], theta, &one, w);
		}

		if(iter/freq == rescnt){
			
			/*Compute duality gap and print from rank 0.*/
			double primal = 0., dual = 0.;
			double primalsum = 0., dualsum = 0.;
			dual = ddot(&len, w, &incx, w, &incx);
			MPI_Allreduce(&dual, &dualsum, 1, MPI_DOUBLE, MPI_SUM, comm);
			/*Compute the dual objective value.*/
			dual = dualsum;
			for(int i = 0; i < n; ++i){
				dual += -2.*alpha[i];
			}

			dual /= 2.;

			/*Compute the primal objective value.*/
			mkl_dcsrmv(&transb, &len, &n, &one, matdesc, &vals[0], &colidx[0], &rowidx[0], &rowidx[1], w, &zero, tildew);
			MPI_Allreduce(tildew, tildewsum, n, MPI_DOUBLE, MPI_SUM, comm);
			for(int i = 0; i < n; ++i){
				primal = 1 - tildewsum[i]*y[i];
				if(primal > 0.)
					primalsum += primal;
			}

			primalsum += dualsum/2.;
			if(rank == 0)
				std::cout << std::scientific << std::setprecision(6) << "D(x) = " << dual << " P(x) = " << primalsum << " Gap = " << primalsum + dual << std::endl;
			if(primalsum + dual <= tol){
				free(alpha); free(G); free(recvG); free(index);
				free(del_alp); free(theta); free(proj_grad); free(grad);
				samprowidx.clear(); sampcolidx.clear(); sampvals.clear(); //sampres.clear();
				MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				MPI_Reduce(&matcpyagg, &matcpymax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				
				if(rank == 0){
					std::cout << "Sampling and extraction time: " << std::setprecision(3) << std::fixed << (matcpymax-grammax)*1000 << " ms" << std::endl;
					std::cout << "Gram Matrix computation time: " << grammax*1000 << " ms" << std::endl;
					std::cout << "Inner loop computation time: " << innermax*1000 << " ms" << std::endl;
					std::cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << std::endl;
				}
				
				return;
			}
			rescnt++;
		}

		if(iter == maxit){
			/*Compute duality gap*/
			
			free(alpha); free(G); free(recvG); free(index);
			free(del_alp); free(theta); free(proj_grad); free(grad);
			samprowidx.clear(); sampcolidx.clear(); sampvals.clear(); //sampres.clear();
			MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&matcpyagg, &matcpymax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			
			if(rank == 0){
				std::cout << "Sampling and extraction time: " << std::setprecision(3) << std::fixed << (matcpymax-grammax)*1000 << " ms" << std::endl;
				std::cout << "Gram Matrix computation time: " << grammax*1000 << " ms" << std::endl;
				std::cout << "Inner loop computation time: " << innermax*1000 << " ms" << std::endl;
				std::cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << std::endl;
			}
			return;
		}


		
		sampcolidx.clear(); sampvals.clear();
		samprowidx[0] = 1;

		std::fill(G, G + (s*s + s), 0.);
		std::fill(recvG, recvG + (s*s + s), 0.);
		//memset(G, 0, sizeof(double)*s*(s + 1));
		//memset(recvG, 0, sizeof(double)*s*(s + 1));
	}
}




int main(int argc, char* argv[])
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

	if(rank == 0)
		std::cout << "Started args parsing" << std::endl;

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

	int sorig = s;
	if(rank == 0)
		std::cout << "Started reading" << std::endl;
	std::string lines = libsvmread(fname, m, n);
	if(rank == 0)
		std::cout << "Finished reading" << std::endl;

	std::vector<int> rowidx, colidx;
	std::vector<double> y, vals;

	int dual_method = 0;
	if(rank == 0)
		std::cout << "Started Parsing" << std::endl;
	parse_lines_to_csr(lines, rowidx, colidx, vals, y, dual_method, m, n);
	if(rank == 0)
		std::cout << "Finished Parsing" << std::endl;
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

	double algst, algstp;
	double *w;
	int ncols = rowidx.size()-1;
	assert(0==Malloc_aligned(double, w, ncols, ALIGN));
	
	int s_lim = 1;
	std::cout << std::setprecision(4) << std::fixed;
	
	s = 1;
	for(int j = 0; j <s_lim; ++j){
		if(rank == 0){
			std::cout << std::endl << std::endl;
			std::cout << "s = " << s << std::endl;
		}
		//cabcd(rowidx, colidx, vals, m, n, y, y.size(), lambda, s, b, maxit, tol, seed, freq, w, comm);
		//MPI_Barrier(MPI_COMM_WORLD);
		algst = MPI_Wtime();
		for(int i = 0; i < niter; ++i){
			//cabcd(localX, n, m, localy, cnts2[rank], lambda, s, b, maxit, tol, seed, freq, w, comm);
			casvm(rowidx, colidx, vals, m, n, y, ncols, lambda, s, maxit, tol, seed, freq, w,comm);
		}
		algstp = MPI_Wtime();
		double algmaxt = 1000*(algstp - algst)/niter;
		double algmin,algmax, algavg;
		MPI_Reduce(&algmaxt, &algmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&algmaxt, &algmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&algmaxt, &algavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0){
			std::cout << std::endl << std::setprecision(3) << std::fixed << "Total CA-SVM time (min): " << algmin << " ms";
			std::cout << std::endl << std::setprecision(3) << "Total CA-SVM time (max): " << algmax << " ms";
			std::cout << std::endl << std::setprecision(3) << "Total CA-SVM time (mean): " << algavg/npes << " ms" << std::endl;
		}
		//MPI_Barrier(MPI_COMM_WORLD);
		s *= 2;
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
