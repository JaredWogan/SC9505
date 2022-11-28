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


int main(int argc, char** argv)
{
  MPI_stuff the_mpi(argc, argv);
  
  std::cout << "Hello from processor " << the_mpi.MyID << " of " 
	    << the_mpi.NProcs << std::endl;

   return 0;
}
