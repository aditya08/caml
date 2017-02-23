#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <mkl.h>
#include <iomanip>

#include "util.h"

int main(int argc, char* argv[]){
	
	MPI_Init(&argc, &argv);
	
	int rank, npes;
	
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char* fname = argv[1];
	int m = atoi(argv[2]);
	int n = atoi(argv[3]);
	int flag = atoi(argv[4]);
	int p = atoi(argv[5]);

	std::string lines = libsvmread(fname, m, n);
	/*
	for(int i = 0; i < 2; ++i)
		for(int j = 0; j < n; ++j)
			std::cout << A[i*n + j] << std::endl;
	*/

	std::vector<int> rowidx, colidx;
	std::vector<double> y, vals;
	parse_lines_to_csr(lines, rowidx, colidx, vals, y);
	std::cout << "rank = " << rank << " ";
	for(int i = 0; i < y.size(); ++i)
		std::cout << y[i] << ' ';
	std::cout << std::endl;
	

	char trans = 'T';
	m = rowidx.size() - 1;
	if(rank == 0)
	std::cout << "nrows = " << m << std::endl;

	double *G, *recvG;
	Malloc_aligned(double, G, n*n, ALIGN);
	Malloc_aligned(double, recvG, n*n, ALIGN);
	
	mkl_dcsrmultd(&trans, &m, &n, &n, &vals[0], &colidx[0], &rowidx[0],  &vals[0], &colidx[0], &rowidx[0], G, &n);
	MPI_Reduce(G, recvG, n*n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
	
	/*
	if(rank == 0){
		for(int i = 0; i < n; ++i) std::cout << std::setprecision(4) << std::fixed << recvG[i*n + i] << " ";
		std::cout << std::endl;
	}*/

	int cursamp = 0, count = 0, b = 3;
	std::vector<int> index;
	srand48(100);
	while(cursamp < b){
		if(((n-count)*drand48()) >= (b - cursamp))
			++count;
		else{
			index.push_back(count);
			if(rank == 0)
				//std::cout << count << ' ';
			++count; ++cursamp;
		}
	}
	//if(rank == 0)
		//std::cout << std::endl;

	std::vector<int> samprowidx(rowidx.size(), 0), sampcolidx;
	std::vector<double> sampvals;


	samprowidx[0] = 1;

	int cidx = 0, rnnz = 0;
	double tval = 0.;

	for(int i = 1; i < rowidx.size(); ++i){
		samprowidx[i] = samprowidx[i-1];
		for(int j = rowidx[i-1]-1; j < rowidx[i]-1; ++j){
			cidx = colidx[j];
			tval = vals[j];
			for(int k = 0; k < b; ++k){
				if(cidx == index[k]+1){
					sampcolidx.push_back(k+1);
					sampvals.push_back(tval);
					samprowidx[i]++;
				}
			}
		}
	}

	/*if(rank == 0){
		std::cout << "rank 0 sampvals length: " << sampvals.size() << " there should be: " << m*b << std::endl;
		std::cout << "rank 0 sampcolidx length: " << sampcolidx.size() << " there should be: " << m*b << std::endl;
		std::cout << "rank 0 samprowidx length: " << samprowidx.size() << " there should be: " << m+1 << std::endl;
		std::cout << "rank 0 samprowidx[length - 1]: " << samprowidx[m] << std::endl;
		
	}*/

	/*
	for(int i = 0; i < sampvals.size(); ++i)
		std::cout << sampvals[i] << ' ';
	std::cout << std::endl;
	for(int i = 0; i < sampvals.size(); ++i)
		std::cout << sampcolidx[i] << ' ';
	std::cout << std::endl;
	for(int i = 0; i < samprowidx.size(); ++i)
		std::cout << samprowidx[i] << ' ';
	std::cout << std::endl;
	*/

	double *sampG = Malloc(double,b*b), *samprecvG = Malloc(double, b*b);
	mkl_dcsrmultd(&trans, &m, &b, &b, &sampvals[0], &sampcolidx[0], &samprowidx[0],  &sampvals[0], &sampcolidx[0], &samprowidx[0], sampG, &b);
	
	MPI_Reduce(sampG, samprecvG, b*b, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
	/*if(rank == 0){
		for(int i = 0; i < b; ++i) std::cout << std::setprecision(4) << std::fixed << samprecvG[i*b + i] << " ";
		std::cout << std::endl;
	}*/
	
	free(G); free(recvG); free(sampG); free(samprecvG);

	MPI_Finalize();
	return 0;
}
