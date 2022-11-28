15-matrix-vector.o: 10-matrix-vector.cpp mv-timings.txt
	mpiCC -O2 -o 15-matrix-vector.o 10-matrix-vector.cpp

mv-timings.txt:
	touch mv-timings.txt

16-serial.o: 11-serial-matrix-vector.cpp
	mpiCC -O2 -o 16-serial.o 11-serial-matrix-vector.cpp

all: 15-matrix-vector.o 16-serial.o

clean:
	rm -rf 15-matrix-vector.o 16-serial.o