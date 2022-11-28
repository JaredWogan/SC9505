15-matrix-matrix-blas.o: 10-matrix-matrix-blas.cpp mmblas-timings.txt
	mpiCC -O2 -o 15-matrix-matrix-blas.o 10-matrix-matrix-blas.cpp -lblas -llapack

mmblas-timings.txt:
	touch mmblas-timings.txt

all: 15-matrix-matrix-blas.o

clean:
	rm -rf 15-matrix-matrix-blas.o