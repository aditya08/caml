#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <mpi.h>

#include "util.h"

void libsvmread(const char* fname, int m, int n, double *A, int leny, double *y){
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

	/* Parse file chunk into the dense vector y and local CSR/CSC matrices based on primal or dual method.
	 * Matrix is already in 1D-column layout. Need to perform All_to_allv for 1D-row.
	 * 
	 * Also, ensure that nnz per rank is roughly load-balanced.
	 * Easiest to just construct CSR/CSC matrices and compare length of values vector.
	 * 
	 * Compare performance with/without LB.
	 * */
	

	MPI_File_close(&in);
	MPI_File_close(&out);
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
