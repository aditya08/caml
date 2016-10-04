#ifndef DEVARAKONDA_deb51010468d4eeb92dc22f494eb9eab
#define DEVARAKONDA_deb51010468d4eeb92dc22f494eb9eab
	
	#include "mpi.h"
	void cabdcd(	double *X,
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
