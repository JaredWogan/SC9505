#include <stdio.h>
#include "mpi.h"

// MPI Stuff
struct mpi_vars {
  int NProcs;
  int MyID;
}; 

struct mpi_vars mpi_start(int argc, char** argv)
{
  struct mpi_vars this_mpi;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &this_mpi.NProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &this_mpi.MyID);

  return this_mpi;
}

// Function to compute the Taylor series of Ln x for x=2
float compute_lnsum(const struct mpi_vars the_mpi, const int N)
{
  float sum = 0, Gsum;

  for(int i = the_mpi.MyID; i < N; i += the_mpi.NProcs)
    if(the_mpi.MyID % 2)
      sum -= (float) 1 / (i + 1);
    else
      sum += (float) 1 / (i + 1);

  MPI_Reduce(&sum,&Gsum,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

  return Gsum;
}

/* See lecture notes for comments */
int main(int argc, char** argv)
{
  struct mpi_vars the_mpi = mpi_start(argc, argv);

  int N;
  if(the_mpi.MyID == 0){
    printf("Please enter the number of terms N -> ");
    fflush(stdout); // Sometimes needed but shouldn't be
    scanf("%d",&N);
  }
  MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
  
  float Gsum = compute_lnsum(the_mpi, N);

  if(the_mpi.MyID == 0)
    printf("An estimate of ln(2) is %f \n",Gsum);

  MPI_Finalize();
  return 0;
}
