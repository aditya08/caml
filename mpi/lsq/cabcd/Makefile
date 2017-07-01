CC = CC
OPT = -O3  -unroll-aggressive -qopt-prefetch
CXXFLAGS = -Wall -g $(OPT) -I../../util 
LDLIBS = -lrt -qopenmp -mkl -L../../util/libutil.a

targets =	cabcd
objects =	cabcd.o

.PHONY : default
default : all

.PHONY : all
all : $(targets)

cabcd : cabcd.o ../../util/libutil.a
	$(CC) -o $@ $^ $(LDLIBS)
cabcd_uniquecols : cabcd_uniquecols.o ../../util/libutil.a
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.cxx cabcd.h
	$(CC) -c $(CXXFLAGS) $(LDLIBS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
