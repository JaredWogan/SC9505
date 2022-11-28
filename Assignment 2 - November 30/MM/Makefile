15-matrix-matrix.o: 10-matrix-matrix.cpp mm-timings.txt
	mpiCC -O2 -o 15-matrix-matrix.o 10-matrix-matrix.cpp

mm-timings.txt:
	touch mm-timings.txt

16-serial-matrix-matrix.o: 11-serial-matrix-matrix.cpp
	mpiCC -O2 -o 16-serial-matrix-matrix.o 11-serial-matrix-matrix.cpp -lblas -llapack

all: 15-matrix-matrix.o 16-serial-matrix-matrix.o

clean:
	rm -rf 15-matrix-matrix.o 16-serial-matrix-matrix.o