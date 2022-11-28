#include <iostream>
#include "mpi.h"

class MPI_stuff 
{
public:
  int NProcs;
  int MyID;

  MPI_stuff(int &argc, char** &argv)
  {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
  }

  ~MPI_stuff()
  {
    MPI_Finalize();
  }
}; 

float compute_lnsum(const MPI_stuff &the_mpi, const int N)
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
  MPI_stuff the_mpi(argc, argv);

  int N;
  if(the_mpi.MyID == 0){
    std::cout << "Please enter the number of terms N -> ";
    std::cin >> N;
  }
  MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);

  float Gsum = compute_lnsum(the_mpi, N);
  
  if(the_mpi.MyID == 0)
    std::cout << "An estimate of ln(2) is " << Gsum << std::endl;

  return 0;
}
