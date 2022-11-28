#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
	
	int rank, size;
	int ierr;

	ierr = MPI_Init(&argc, &argv);
	
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("Hello, world from task %d of %d!\n",rank,size); 

	MPI_Finalize();

	return ierr;
}
