FT = 250 # frames time in milliseconds
F = 1000 # frames to run for performance data
S = 24 # size of the grid
W = 800 # width of the window
H = 800 # height of the window

TRIALS = $(shell seq 1 1 5)

ITERS_MIN = 1000
ITERS_MAX = 10000
ITERS_STEP = 1000
ITERS = 100 500 1000 2500 5000 10000

GRIDS_MIN = 100
GRIDS_MAX = 1000
GRIDS_STEP = 100
GRIDS = 108 216 324 504 756 1008 2016

PROCS = 4 9 12 16 18

gol.o: gol-pg.cpp
	mpiCC -O2 -o gol.o gol-pg.cpp -lSDL2

perf.o: perf-pg.cpp
	mpiCC -O2 -o perf.o perf-pg.cpp

all: gol.o perf.o

run: gol.o gol-pg.cpp
	mpirun -n 4 ./gol.o $(FT) $(S) $(S) $(W) $(H) 10 10 1 1

perf: perf.o perf-pg.cpp
	mpirun -n 4 ./perf.o $(F) $(S) $(S) 0 0

data: perf.o perf-pg.cpp
	echo "size_x, size_y, threads, iterations, total_runtime, average_frame_time, comm_time" > data.csv
	
	for s in $(GRIDS); do \
		for i in $(ITERS); do \
			for p in $(PROCS); do \
				for t in $(TRIALS); do \
					mpirun --oversubscribe -n $$p ./perf.o $$i $$s $$s 0 1; \
				done; \
			done; \
		done; \
		echo "Finished grids of size $$s x $$s"; \
	done

clean:
	rm -rf gol.o perf.o