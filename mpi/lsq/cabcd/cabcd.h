#ifndef DEVARAKONDA_dcb882eca214420ca7527adf95741bb5
#define DEVARAKONDA_dcb882eca214420ca7527adf95741bb5
	
	#include "mpi.h"
	#include <vector>

	void cabcd(	std::vector<int> &rowidx,
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
				MPI_Comm comm);

#endif
