15-poisson.o: 10-poisson.cpp lp-timings.txt lp-data.txt
	mpiCC -O2 -o 15-poisson.o 10-poisson.cpp -lblas -llapack

lp-timings.txt:
	touch lp-timings.txt

lp-data.txt:
	touch lp-data.txt

all: 15-poisson.o

clean:
	rm -rf 15-poisson.o