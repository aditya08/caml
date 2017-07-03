#ifndef DEVARAKONDA_3f44bbc454b411e7b114b2f933d5fe66
#define DEVARAKONDA_3f44bbc454b411e7b114b2f933d5fe66
	
	#include "mpi.h"
	#include <vector>

	void casvm(		std::vector<int> &rowidx,
					std::vector<int> &colidx,
					std::vector<double> &vals,
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
					MPI_Comm comm);

inline double sign(double x){return (x > 0.) ? 1. : ((x < 0.) ? -1. : 0.);}
inline double max(double x, double y){return (x > y) ? x : y;}
#endif
