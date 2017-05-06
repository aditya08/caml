#ifndef DEVARAKONDA_08355951500c42aa93fc19d31d28cba3
#define DEVARAKONDA_08355951500c42aa93fc19d31d28cba3
	
	#include "mpi.h"
	#include <vector>

	void calasso(	std::vector<int> &rowidx,
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
					double *v,
					MPI_Comm comm);

inline double sign(double x){return (x > 0.) ? 1. : ((x < 0.) ? -1. : 0.);}
inline double max(double x, double y){return (x > y) ? x : y;}
#endif
