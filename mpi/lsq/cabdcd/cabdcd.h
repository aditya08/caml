#ifndef DEVARAKONDA_deb51010468d4eeb92dc22f494eb9eab
#define DEVARAKONDA_deb51010468d4eeb92dc22f494eb9eab
	
	#include "mpi.h"
	#include <vector>
	void cabdcd(	std::vector<int> &rowidx,
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
