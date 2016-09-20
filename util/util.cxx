#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"

void libsvmread(const char* fname, int m, int n, double *A, int leny, double *y){
	int i = 0, idx = 0;
	std::ifstream file(fname);
	std::string line;
	std::size_t pos;

	assert(0!=file.is_open());
	std::cout << "about to read from file" << std::endl;
	
	while(file){
		file >> line;
		pos = line.find(":");
		if(pos != std::string::npos){
			//std::cout << "column = " << atoi(line.substr(0,pos).c_str()) << std::endl;
			idx = n*(i-1) + (atoi(line.substr(0,pos).c_str())-1);
			//std::cout << "idx = " << idx << std::endl;
			A[idx] = atof(line.substr(pos+1).c_str());
		}
		else{
			y[i] = atof(line.c_str());
			++i;
		}
	}

	file.close();
}

void staticLB_1d(int m, int n, int npes, int flag, int *cnts, int *displs, int *cnts2, int *displs2)
{
	int mm, nn;
	
	if (flag){
		mm = n;
		nn = m;
	}
	else{
		mm = m;
		nn = n;
	}

	for (int i = 0; i < mm%npes; ++i)
	{
		cnts[i] = (mm/npes + 1)*nn;
		cnts2[i] = (mm/npes + 1);
		displs[i] = (i*(mm/npes + 1))*nn;
		displs2[i] = (i*(mm/npes + 1));
		//std::cout << cnts[i] << std::endl;
	}

	for (int i = mm%npes; i < npes; ++i)
	{
		cnts[i] = (mm/npes)*nn;
		cnts2[i] = mm/npes;
		displs[i] = ((mm%npes)*(mm/npes + 1) + (i - mm%npes)*mm/npes)*nn;
		displs2[i] = ((mm%npes)*(mm/npes + 1) + (i - mm%npes)*mm/npes);
		//std::cout << cnts[i] << std::endl;
	}
}
