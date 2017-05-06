#ifndef DEVARAKONDA_659e44e464ee4dd885d83d5321cd1326
#define DEVARAKONDA_659e44e464ee4dd885d83d5321cd1326

#define ALIGN 32

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Malloc_aligned(type, ptr, n, align) posix_memalign((void**)&ptr, align, sizeof(type)*n)

#include<vector>


inline int compare_idx(const void *a, const void *b){
	return ( *(int *)a - *(int *)b );
}

std::string libsvmread(const char*, int, int);

/*
Use this routine to transpose the input matrix and write it out.
If you have a short-wide matrix, parallel File I/O performs implict 1D-row. 
If you need 1D-col, then you need to transpose (i.e. Alltoallv), but this throws off load-balancing.
Best solution is probably to create a new file which stores the transpose of the matrix.
*/
void libsvmwrite(std::vector<int>&, std::vector<int>&, std::vector<double>&, std::vector<double>&, int, int, const char*);


void parse_lines_to_csr(std::string, std::vector<int>&, std::vector<int>&, std::vector<double>&, std::vector<double>&, int, int, int);
void staticLB_1d(int, int, int, int, int*, int*, int*, int*);
//void staticLB_1drow(int, int, int, int**, int**);

#endif
