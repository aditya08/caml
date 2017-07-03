CC = CC
OPT = -O3  -unroll-aggressive -qopt-prefetch
CXXFLAGS = -Wall -g $(OPT) -I../util 
LDLIBS = -lrt -qopenmp -mkl=sequential -L../util/libutil.a

targets =	casvm
objects =	casvm.o

.PHONY : default
default : all

.PHONY : all
all : $(targets)

casvm : casvm.o ../util/libutil.a
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.cxx casvm.h
	$(CC) -c $(CXXFLAGS) $(LDLIBS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
