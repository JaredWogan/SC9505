all:
	cd ./Serial
	make all
	cd ../ParallelSeq
	make all
	cd ../ParallelGrid
	make all
	cd ..

pdata:
	make -C ./ParallelSeq data
	make -C ./ParallelGrid data

data:
	make -C ./Serial data
	make -C ./ParallelSeq data
	make -C ./ParallelGrid data

clean:
	rm -rf ./Serial/gol.o ./Serial/perf.o
	rm -rf ./ParallelSeq/gol.o ./ParallelSeq/perf.o
	rm -rf ./ParallelGrid/gol.o ./ParallelGrid/perf.o