#ifndef DEVARAKONDA_659e44e464ee4dd885d83d5321cd1326
#define DEVARAKONDA_659e44e464ee4dd885d83d5321cd1326

#define ALIGN 32

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Malloc_aligned(type, ptr, n, align) posix_memalign((void**)&ptr, align, sizeof(type)*n)

void libsvmread(const char*, int, int, double*, int, double*);
void staticLB_1d(int, int, int, int, int*, int*, int*, int*);
//void staticLB_1drow(int, int, int, int**, int**);

#endif