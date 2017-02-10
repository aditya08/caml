#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <mpi.h>

#include "util.h"

std::string libsvmread(const char* fname, int m, int n){
	int i = 0, idx = 0;

	int rank;
	int npes;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);

	MPI_Offset end;
	MPI_Offset fsize;
	MPI_Offset start;

	int localrdsize;
	int overlap = n*100;
	char *rdchars;


	MPI_File in;
	int ierr = MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &in);
	assert(!ierr);

	MPI_File_get_size(in, &fsize);

	localrdsize = fsize/npes;
	start = rank*localrdsize;
	end = localrdsize + start - 1;

	if(rank == npes - 1) end = fsize - 1;

	if(rank != npes - 1) end += overlap;

	localrdsize = end - start + 1;

	rdchars = Malloc(char,localrdsize+1);

	ierr = MPI_File_read_at_all(in, start, rdchars, localrdsize, MPI_CHAR, MPI_STATUS_IGNORE);
	assert(!ierr);
	MPI_File_close(&in);
	
	int lstart = 0, lend = localrdsize - 1;
	if(rank != 0){
		while(rdchars[lstart] != '\n')
			lstart++;
		lstart++;
	}

	if(rank != npes - 1){
		lend-= overlap;
		while(rdchars[lend] != '\n')
			lend++;
	}

	localrdsize = lend - lstart + 1;
	//rdchars[localrdsize] = '\0';

	std::string lines(rdchars + lstart, localrdsize);
	
	//int currank = 0;
	//while(currank < npes){
	//	if(currank == rank){
	//		std::cout << lines;
	//		fflush(stdout);
	//	}
	//	currank++;
	//	MPI_Barrier(MPI_COMM_WORLD);

	//}
	//
	MPI_File out;
	ierr = MPI_File_open(MPI_COMM_WORLD, "out.txt", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &out);
	
	MPI_File_write_at_all(out, (MPI_Offset)(start + (MPI_Offset)lstart), &rdchars[lstart], localrdsize, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&out);

	return lines;

	
}

void parse_lines_to_csr(std::string lines, std::vector<int> &rowidx, std::vector<int> &colidx, std::vector<double> &vals, std::vector<double> &y, std::vector<int> &col_offsets){
	/* Parse file chunk into the dense vector y and local 3-array CSR matrices based.
	 * Matrix is already in 1D-column layout (Good for primal method). Need to perform All_to_allv for 1D-row (good for dual method).
	 * 
	 * Also, ensure that nnz per rank is roughly load-balanced.
	 * Easiest to just construct CSRmatrices and compare length of vals vector.
	 * 
	 * Load Balance if needed. Compare performance with/without LB.
	 * */

	int rank, npes;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);

	int nrows = 0, nnz = 0;
	std::size_t curl = 0, curw = 0, curc = 0;
	std::string line;
	std::string word;
	std::string tmp;

	rowidx.push_back(1);

	while( (curl = lines.find('\n')) != std::string::npos ){
		line = lines.substr(0,curl);
		//std::cout << line << std::endl;
		lines.erase(0,curl + 1);
		
		while( line.length() != 0){
			curw = (line.find(' ') != std::string::npos) ? (line.find(' ')) : line.length();
			word = line.substr(0, curw);
			curc = word.find(':');
			tmp = word.substr(0,curc);
			if(curc == std::string::npos){
				y.push_back(atof(tmp.c_str()));
				word.erase(0, curc + 1);
				//std::cout << y.back() << ' ';
			}
			else{
				//parse to one-indexed, 3-array CSR matrix
				colidx.push_back(atoi(tmp.c_str()));
				word.erase(0, curc + 1);
				tmp = word.substr(0,word.length());
				vals.push_back(atof(tmp.c_str()));
				nnz++;
				//std::cout << colidx.back() << ':' << vals.back() << ' ';
			}

			line.erase(0,curw + 1);
		}
		//std::cout << std::endl;
		rowidx.push_back(nnz + rowidx[nrows]);
		nnz = 0;
		nrows++;
	}
	nnz = vals.size();
	std::cout << "Processor " << rank << " has " << nrows << " rows with " << nnz << " nnzs." << std::endl;
	int *nnz_cnts;
	if(rank == 0) Malloc(int, npes);
	MPI_Gather(&nnz, 1, MPI_INT, nnz_cnts, npes, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank == 0){
	//Print statistics on the load per processor. Used to determine the need for load balancing.
		int sum = 0;
		int max = nnz_cnts[0];
		int min = nnz_cnts[0];
		std::cout << "Processors:\t";
		for(int i = 0; i < npes; ++i){
			std::cout << i << '\t';
			sum += nnz_cnts[i];
			max = (max < nnz_cnts[i]) ? nnz_cnts[i] : max;
			min = (min > nnz_cnts[i]) ? nnz_cnts[i] : min;
		}
		std::cout << "\nNNZs:\t"
		for(int i = 0; i < npes; ++i)
			std::cout << nnz_cnts[i] << '\t';
		std::cout << "Max NNZ: " << max << std::endl;
		std::cout << "Min NNZ: " << min << std::endl;
		std::cout << "Avg NNZ: " << (float)sum/npes << std::endl;
	}
}

void parse_lines_to_csc(){

}

void staticLB_1d(int m, int n, int npes, int flag, int *cnts, int *displs, int *cnts2, int *displs2)
{
	int mm, nn;
	
	//1D-block row layout
	if (flag){
		mm = n;
		nn = m;
	}
	//1D-block column layout
	else{
		mm = m;
		nn = n;
	}

	for (int i = 0; i < mm%npes; ++i)
	{
		cnts[i] = ((mm/npes) + 1)*nn;
		cnts2[i] = ((mm/npes) + 1);
		displs[i] = (i*(mm/npes + 1))*nn;
		displs2[i] = (i*(mm/npes + 1));
		//std::cout << cnts[i] << std::endl;
	}

	for (int i = mm%npes; i < npes; ++i)
	{
		cnts[i] = (mm/npes)*nn;
		cnts2[i] = mm/npes;
		displs[i] = (((mm%npes)*(mm/npes + 1)) + ((i - mm%npes)*(mm/npes)))*nn;
		displs2[i] = (((mm%npes)*(mm/npes + 1)) + ((i - mm%npes)*(mm/npes)));
		//std::cout << displs2[i] << std::endl;
	}
}
