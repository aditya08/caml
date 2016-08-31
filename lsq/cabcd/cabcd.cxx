#include <mlk.h>
#include "mpi.h"
#include "cabcd.h"

void cabcd(	double *X,	//input args
		int m,
		int n,
		double *y,
		double *lambda,
		int s,
		int b,
		int maxit,
		double tol,
		int seed,
		int freq,
		double **w)	//output arg: allocated in function.
{
	double *alpha, *obj_err, *sol_err, 
	double del_w;

	double *G, *Xsamp, 

	//sampling tuning choices: randomly permute data matrix during I/O. after I/O. randomly select a column at a time.

	
}



