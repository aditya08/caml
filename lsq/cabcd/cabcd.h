#ifndef DEVARAKONDA_dcb882eca214420ca7527adf95741bb5
#define DEVARAKONDA_dcb882eca214420ca7527adf95741bb5
	
	#include "mpi.h"
	void cabcd(	double *X,
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
				MPI_Comm comm);

#endif
