#include <string>
#include <string.h>
#include <istream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <mpi.h>


#include "util.h"

std::string libsvmread(const char* fname, int m, int n){
	int i = 0, idx = 0;

	int rank;
	int npes;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);

	MPI_Offset end;
	MPI_Offset fsize;
	MPI_Offset start;

	long int localrdsize = 0;
	long int overlap = 100*n;
	char *rdchars;

	double tread = MPI_Wtime();
	MPI_File in;
	long int ierr = MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &in);
	assert(!ierr);

	MPI_File_get_size(in, &fsize);
	//fsize--;

	localrdsize = fsize/npes;
	start = rank*localrdsize;
	end = localrdsize + start - 1;

	if(rank == npes - 1) end = fsize;

	if(rank != npes - 1) end += overlap;

	//if(end > fsize - 1) end = fsize - 1;

	localrdsize = end - start + 1;
	
	//if(rank == 0)
	//	std::cout << "localrdsize: " << localrdsize + 1  << " " << fsize << " " << start  << " "  << end << std::endl;
	rdchars = Malloc(char,localrdsize+1);
	memset(rdchars, 0, sizeof(char)*(localrdsize+1));
//	std::cout << "start rank: " << rank << " " <<  start << " " << end << " " << localrdsize << std::endl;

	ierr = MPI_File_read_at_all(in, start, rdchars, localrdsize, MPI_CHAR, MPI_STATUS_IGNORE);
	assert(!ierr);
	MPI_File_close(&in);

	int lstart = 0, lend = localrdsize - 1;
	if(rank != 0){
		//if(rank == npes - 1)
		//	std::cout << "last procs's first char: " << rdchars[0]<< std::endl;
		while(rdchars[lstart] != '\n' && lstart < lend)
			lstart++;
		lstart++;
	}

	if(rank != npes - 1){
		lend-= overlap;
		if(rdchars[lend] == '\n')
			lend++;
		while(rdchars[lend] != '\n' && lend < localrdsize - 1)
			lend++;
	}


	//std::cout << rank  << " f " << start << " " << localrdsize << std::endl;
	localrdsize = lend - lstart + 1;
	//rdchars[localrdsize] = '\0';
	//std::cout << "started parsing rank: " << rank << " " <<  lstart << " " << lend << " " << localrdsize << std::endl;
	//std::cout << rank << " s " << lstart << " " << localrdsize << std::endl;
	std::string lines(rdchars + lstart, localrdsize);
	tread = MPI_Wtime() - tread;
	double tmax;
	MPI_Reduce(&tread, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if(rank == 0)
		std::cout << "Max read time: " << tmax*1e3 << " ms" << std::endl;

	//int currank = 0;
	//while(currank < npes){
	//	if(currank == rank){
	//		std::cout << lines;
	//		fflush(stdout);
	//	}
	//	currank++;
	//	MPI_Barrier(MPI_COMM_WORLD);

	//}
	//
	//
	//
	/*
		MPI_File out;
		ierr = MPI_File_open(MPI_COMM_WORLD, "out.txt", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &out);

		MPI_File_write_at_all(out, (MPI_Offset)(start + (MPI_Offset)lstart), lines.c_str(), localrdsize, MPI_CHAR, MPI_STATUS_IGNORE);
		MPI_File_close(&out);
	*/
	return lines;
}


void libsvmwrite(std::vector<int> &rowidx, std::vector<int> &colidx, std::vector<double> &vals, std::vector<double> &y, int m, int n, const char* fname){
	int lowptr, hiptr;
	std::string lines;
	std::stringstream ss;
	for(int i = 1; i < rowidx.size(); ++i){
		lowptr = rowidx[i-1]-1;
		hiptr = rowidx[i]-1;
		for(int j = lowptr; j < hiptr; ++j){
			if(j == lowptr){
				if(i-1 < m)
					ss << y[i-1];
				else
					ss << "nan";
				lines += ss.str();
				ss.str("");
			}
			ss << ' ' << colidx[j] << ':' << vals[j];
			lines += ss.str();
			ss.str("");
		}
		if(lines.length() != 0)
			lines += '\n';
	}

	MPI_File out;
	long int ierr = MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &out);

	MPI_File_write_at_all(out, 0, lines.c_str(), lines.length(), MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&out);
	
	return;
}

/* Parse file chunk into the dense vector y and local 3-array CSR matrices based.
 * Matrix is already in 1D-column layout (Good for primal method). Need to perform All_to_allv for 1D-row (good for dual method).
 *
 * Also, ensure that nnz per rank is roughly load-balanced.
 * Easiest to just construct CSRmatrices and compare length of vals vector.
 *
 * Load Balance if needed. Compare performance with/without LB.
 * */
void parse_lines_to_csr(std::string lines, std::vector<int> &rowidx, std::vector<int> &colidx, std::vector<double> &vals, std::vector<double> &y, int dual_method, int m, int n){

	int rank, npes;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);

	//std::cout << "Starting parse..." << std::endl;

	double tparse = MPI_Wtime();

	int nrows = 0, nnz = 0, ncols = 0, col = 0;
	std::size_t curl = 0, curw = 0, curc = 0;
	std::string line;
	std::string word;
	std::string tmp;

	rowidx.push_back(1);
	//std::cout << "original " << lines << std::endl << std::endl;

	std::stringstream strm_line(lines);
	while(std::getline(strm_line, word, '\n') && !strm_line.eof()){
		//std::cout << word << std::endl;
		word += ' ';
		std::stringstream toks(word);
		while(std::getline(toks, tmp, ' ') && !toks.eof()){
			curl = tmp.find(':');
			if(curl == std::string::npos && tmp.compare("nan") != 0 && tmp.length() != 0){
				//std::cout << tmp.c_str() << std::endl;
				y.push_back(atof(tmp.c_str()));
				//std::cout << atof(tmp.c_str()) << std::endl;
				//std::cout << tmp << ' ' << tmp.length() <<  '\n';
			}
			else if(tmp.length() != 0 && tmp.compare("nan") != 0){
				word = tmp.substr(0,curl);
				colidx.push_back(atoi(word.c_str()));
				//ncols = (colidx.back() > ncols) ? colidx.back() : ncols;
				word = tmp.substr(curl+1,tmp.length());
				vals.push_back(atof(word.c_str()));
				nnz++;
				//std::cout << tmp.substr(0,curl) << ':' << tmp.substr(curl+1, tmp.length()) << ' ';
			}
		}
		//if(nnz == 0){
		//	rowidx.push_back(1 + rowidx[nrows]);
		//	colidx.push_back(1);
		//	vals.push_back(0.);
		//}
		//else{
			rowidx.push_back(nnz + rowidx[nrows]);
		//}
		nnz = 0;
		nrows++;
		if(y.size() != nrows && y.size() < n)
		{
			y.push_back(0.);
		}
		//std::cout << std::endl;
	}
	//std::cout << "nnz = " << rowidx.back() << std::endl;

	nnz = (int)vals.size();
	tparse = MPI_Wtime() - tparse;
	double tmax;

	MPI_Reduce(&tparse, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(rank == 0) std::cout << "Max parse time " << tmax*1e3 << " ms" << std::endl;
	
	if(dual_method){
		double ttrans = MPI_Wtime();
		
		std::vector<double> send_vals;
		std::vector<int> send_colidx;
		std::vector<int> send_rowidx(1,1);
		std::vector<int> send_cnts(npes, 0);
		std::vector<int> send_displ(npes+1, 0);
		std::vector<int> pcols(npes, 0);
		std::vector<int> offsets(npes+ 1, 0);
		size_t avg  = n/npes;
		size_t rem = n % npes;
		offsets[0] = 1;

		//std::cout << "Ncols = " << n << " Avg = " << avg << " rem = " << rem << std::endl;
 
		//compute (inclusive) prefix sum in offsets.
		//if(rank == 0)
		//std::cout << "Offsets ";
		
		/*
		for (size_t i = 0; i < npes; i++){
			if(i < rem){
				pcols[i] = avg + 1;
				offsets[i+1] = offsets[i] + avg + 1;
			}
			else{
				pcols[i] = avg;
				offsets[i+1] = offsets[i] + avg;
			}

			//if(rank == 0)
			//	std::cout << offsets[i] << ' ';
		}
		*/
		//if(rank == 0)
		//std::cout << offsets.back() << std::endl;

		/*begin 1D-column partitioning by filling send_* buffers for each processor.
			Loop through vals and colidx to fill send_* buffers according to offesets.
		Alltoallv appropriate here.
		*/

		/*
		int tmpcolidx, tmprowidx = 1, rownnz = 0;
		
		std::vector<int> send_rowcnts(npes, 1), send_rowdispl(npes+1, 0);
		for (size_t i = 0; i < npes; i++) {
			for(int j = 1; j < rowidx.size(); ++j){
				for(int k = rowidx[j-1]-1; k < rowidx[j]-1; ++k){
					tmpcolidx = colidx[k];
					if(tmpcolidx >= offsets[i] && tmpcolidx < offsets[i+1]){
						send_vals.push_back(vals[k]);
						send_colidx.push_back(colidx[k] - offsets[i] + 1);
						send_cnts[i]++;
						rownnz++;
					}
				}
				send_rowidx.push_back(rownnz + send_rowidx.back());
				rownnz = 0;
				send_rowcnts[i]++;
			}
			send_rowidx.push_back(1);
			//std::cout << std::endl;
			send_rowdispl[i+1] = send_rowcnts[i] + send_rowdispl[i];
			send_displ[i+1] = send_cnts[i] + send_displ[i];
		}


		send_displ.pop_back();
		std::vector<int> total_cnts(npes, 0);
		//Call MPI_Alltoall to get partial send cnts to get recv displs.

		MPI_Alltoall(&send_cnts[0], 1, MPI_INT, &total_cnts[0], 1, MPI_INT, MPI_COMM_WORLD);
		*/
		//for(int i = 0; i < npes; ++i)
		//	std::cout << "rank " << rank << " recv_cnts[" << i << "] = " << total_cnts[i];
		//std::cout << std::endl;

		//MPI_Allreduce(&send_cnts[0], &total_cnts[0], npes, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		
		/*
		std::vector<int> recv_displ(npes, 0);
		int total = total_cnts[0];
		for (size_t i = 1; i < npes; i++) {
			recv_displ[i] = total_cnts[i-1] + recv_displ[i-1];
			total += total_cnts[i];
		}
		std::vector<double> recv_vals(total, 0.);
		std::vector<int> recv_cols(total, 0.);
		MPI_Alltoallv(&send_vals[0], &send_cnts[0], &send_displ[0], MPI_DOUBLE, &recv_vals[0], &total_cnts[0], &recv_displ[0], MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Alltoallv(&send_colidx[0], &send_cnts[0], &send_displ[0], MPI_INT, &recv_cols[0], &total_cnts[0], &recv_displ[0], MPI_INT, MPI_COMM_WORLD);
		
		
		MPI_Alltoall(&send_rowcnts[0], 1, MPI_INT, &total_cnts[0], 1, MPI_INT, MPI_COMM_WORLD);
		total = total_cnts[0];
		for (size_t i = 1; i < npes; i++) {
			recv_displ[i] = total_cnts[i-1] + recv_displ[i-1];
			total += total_cnts[i];
		}
		std::vector<int> recv_rowidx(total, 0);
		MPI_Alltoallv(&send_rowidx[0], &send_rowcnts[0], &send_rowdispl[0], MPI_INT, &recv_rowidx[0], &total_cnts[0], &recv_displ[0], MPI_INT, MPI_COMM_WORLD);
		*/
		/*re-construct rowidx vector (ASSUMPTION: colidxs are in increasing order, so row_end/newrow_begin are easy to find.)
		*/
		/*
		int prevcol = recv_cols[0];
		int curcol = 0, cnt_rownnz = 2;
		std::vector<int> new_rowidx(1,1);
		//std::cout << new_rowidx.back() << ' ';
		for(int i = 1; i < recv_rowidx.size(); ++i){
			if(recv_rowidx[i] != 1){
				new_rowidx.push_back(new_rowidx.back() + (recv_rowidx[i] - recv_rowidx[i-1]));
				//std::cout << new_rowidx.back() << ' ';
			}
		}
		*/
		//std::cout << std::endl;
		//recv_rowidx.clear();
		//for(int i = 0; i < new_rowidx.size(); ++i){
		//	std::cout << new_rowidx[i] << ' ';
		//}
		//std::cout << std::endl;
		//nnz = new_rowidx.back() - 1;
		//std::cout << "nnz = " << new_rowidx.back() << std::endl;
		//Need to transpose matrix since mkl_dcsrmult only allows X^T*X and not XX^T.
			//**TODO: If mkl_dcsrmult eventually allows XX^T, then we can avoid transpose.
			//**TODO: If mkl also supports CSC (i.e. mkl_dcscmult), then we can store in CSC.

		//for(int i = 0; i < recv_cols.size(); ++i)
		//	std::cout << recv_cols[i] << ':' << recv_vals[i] << ' ';
		//std::cout << std::endl;


		/*
		for(int i = 1; i < new_rowidx.size(); ++i){
			for(int j = new_rowidx[i-1]-1; j < new_rowidx[i]-1; ++j){
				std::cout << recv_cols[j] << ':' << recv_vals[j] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		*/

		std::vector<int> trans_cols;
		std::vector<double> trans_vals;
		int rowptr = 1;
		std::vector<int> new_rowidx; 
		new_rowidx.push_back(1);
		int rownnz = 0;
		for(int i = 0; i < n; ++i){
			for(int j = 1;  j < rowidx.size(); ++j){
				for(int k = rowidx[j-1]-1; k < rowidx[j]-1; ++k){
					if(colidx[k] == i+1){
						trans_cols.push_back(j);
						trans_vals.push_back(vals[k]);
						rownnz++;
					}
				}
			}
			new_rowidx.push_back(rownnz + new_rowidx.back());
			rownnz = 0;
		}

		//recv_cols = trans_cols;
		//recv_vals = trans_vals;
		
		colidx = trans_cols;
		vals = trans_vals;
		rowidx = new_rowidx;

		trans_cols.clear(); trans_vals.clear(); new_rowidx.clear();
		//std::cout << "rowidx size = " << rowidx.size() << " actual numrows should be = " << pcols[rank] << std::endl;
		/*
		for(int i = 1; i < rowidx.size(); ++i){
			for(int j = rowidx[i-1]-1; j < rowidx[i]-1; ++j){
				std::cout << colidx[j] << ':' << vals[j] << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		*/
		//Matrix is now in 1D-col layout, just need to Allgather the labels vector, y.
		//std::vector<int> ylen_comm(npes, 0);
		//std::vector<int> ydispl(npes, 0);
		//int sendlen = y.size();
		//MPI_Allgather(&sendlen, 1, MPI_INT, &ylen_comm[0], 1, MPI_INT, MPI_COMM_WORLD);

		//nrows = ylen_comm[0];
		//if(rank == 0)
		//	std::cout << "ydispl " << ydispl[0];
		/*
		for(int i = 1; i < npes; ++i){
			ydispl[i] = ylen_comm[i-1] + ydispl[i-1];
			nrows += ylen_comm[i];
		//	if(rank == 0)
		//	std::cout << " " << ydispl[i];
		}
		*/
		//if(rank == 0)
		//	std::cout << std::endl;

		/*
		std::vector<double> new_y(nrows, 0.);
		MPI_Allgatherv(&y[0], sendlen, MPI_DOUBLE, &new_y[0], &ylen_comm[0], &ydispl[0], MPI_DOUBLE, MPI_COMM_WORLD);
		*/

		//Need to copy recv_vals, _cols, new_rowidx and new_y into the input std::vectors rowidx, colidx, vals, and y.
		//rowidx = new_rowidx;
		//colidx = recv_cols;
		//vals = recv_vals;
		//y = new_y;
		//new_rowidx.clear(); recv_cols.clear(); recv_vals.clear(); new_y.clear();
	
		ttrans = MPI_Wtime() - ttrans;
		double ttr;

		MPI_Reduce(&ttrans, &ttr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		if(rank == 0) std::cout << "Max transpose time " << ttrans*1e3 << " ms" << std::endl;
	
	}



	double tstat = MPI_Wtime();

	//std::cout << "Processor " << rank << " has " << nrows << " rows with " << nnz << " nnzs." << std::endl;
	int *nnz_cnts = NULL;
	if(rank == 0) nnz_cnts =  Malloc(int, npes);
	MPI_Gather((void*)&nnz, 1, MPI_INT, (void*)nnz_cnts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int sum, max, min, med;

	if(rank == 0){
		sum = 0;
		max = nnz_cnts[0];
		min = nnz_cnts[0];
	//Print statistics on the load per processor. Used to determine the need for load balancing.
		//std::cout << "Procs:\t";
		for(int i = 0; i < npes; ++i){
		//	std::cout << i << '\t';
			sum += nnz_cnts[i];
			max = (max < nnz_cnts[i]) ? nnz_cnts[i] : max;
			min = (min > nnz_cnts[i]) ? nnz_cnts[i] : min;
		}
		/*
		std::cout << "\nNNZs:\t";
		for(int i = 0; i < npes; ++i)
			std::cout << nnz_cnts[i] << '\t';
		std::cout << std::endl << std::endl;;
		*/
		qsort(nnz_cnts, npes, sizeof(int), compare_idx);
		med = (npes % 2 == 0) ? ((nnz_cnts[(npes-1)/2] + nnz_cnts[(npes-1)/2 + 1])/2) : nnz_cnts[npes/2];
	}

	tstat = MPI_Wtime() - tstat;
	MPI_Reduce(&tstat, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(rank == 0){

		std::cout << "Max stats time " << tstat*1e3 << " ms" << std::endl;

		std::cout << std::endl << std::endl;

		std::cout << "//*********************************//" << std::endl;
		std::cout << "//\tMin NNZ: " << min  << std::endl;
		std::cout << "//\tMax NNZ: " << max << " (Load Imbalance: " << std::setprecision(2) << std::fixed << (float)(max - (sum/npes))/(sum/npes)*100 << "%)" << std::endl;
		std::cout << "//\tMed NNZ: " << med <<  std::endl;
		std::cout << "//\tAvg NNZ: " << (float)sum/npes << std::endl;
		std::cout << "//\tTotal NNZ: " << (float)sum << std::endl;
		std::cout << "//*********************************//" << std::endl;
		std::cout << std::endl << std::endl;
	}
}
