CC = CC
OPT = -O3  -unroll-aggressive -qopt-prefetch
CXXFLAGS = -fPIC -Wall -g $(OPT) -mkl=sequential#-I$(MKL_INC)
# librt is needed for clock_gettime
LDLIBS = -lrt -qopenmp -mkl=sequential

targets = bench_svmread
objects = bench_svmread.o util.o

.PHONY : default
default : all

.PHONY : all
all : $(targets)

bench_svmread : bench_svmread.o util.o
	$(CC) -o $@ $^ $(LDLIBS) -shared
	ar -crf libutil.a util.o 

%.o : %.cxx util.h
	$(CC) -c $(CXXFLAGS) $<

libutil : util.o
	$(CC) -o $@ $^ $(LDLIBS) -shared
	ar -rv libutil.so libutil

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
