#include <iostream>
#include <fstream>
#include <sstream>
#define BOOST_DISABLE_ASSERTS      // Only do this after you are sure the code works!
#include "boost/multi_array.hpp"
#include "boost/shared_ptr.hpp"
#include "mpi.h"
// Solve Poisson's equations using the Jacobi method, with MPI parallization
// dividing the grid into a 1D array of processors 
// Compile with mpiCC -O3 PoissonJacobi1DMPI.cpp

#define N  1000
#define Tol  0.0001
typedef boost::multi_array<double,2> Doub2D;

using namespace std;

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


// breakup array size N in 1D into parts for each processor
int compute_my_size(MPI_stuff &the_mpi)
{
  int remainder = (N - 1) % the_mpi.NProcs;
  int size = (N - 1 - remainder)/the_mpi.NProcs;
  if(the_mpi.MyID < remainder)  // extra rows added for MyID < remainder 
    size = size + 2;
  else
    size = size + 1;
  return size;
}


void initialize(Doub2D &old, Doub2D &now, int size, MPI_stuff &the_mpi)
{ /* Assume interior is already initialized, just need to set boundaries */
  /* B.Cs set to a constant, 1 here */

  for(int i = 0; i < size + 1; i++)
    now[i][0] = now[i][N] = old[i][0] = old[i][N] = 1;
  if(the_mpi.MyID == 0)
    for(int j = 1; j < N; j++)
      now[0][j] = old[0][j] = 1;
  if(the_mpi.MyID == the_mpi.NProcs - 1)
    for(int j = 1; j < N; j++)
      now[size][j] = old[size][j] = 1;
}


double iteration(Doub2D &old, Doub2D &now, int start, int finish)
{   /* Jacobi iteration */
    double maxerr = 0;
    
    for(int i = start; i < finish; i++) {
       for(int j = 1; j < N; j++){
	 now[i][j] = 0.25*(old[i+1][j] + old[i-1][j] +
                            old[i][j+1] + old[i][j-1]);

	 maxerr = fmax(maxerr, fabs(now[i][j] - old[i][j]));
       }
    }
    return (maxerr);
}


/* Output result */
void output(Doub2D &now, int size, MPI_stuff &the_mpi)
{
  std::ostringstream filename;
  filename << "Solution" << the_mpi.MyID << ".Txt";
  std::ofstream fp(filename.str());

  if(the_mpi.MyID == 0) {
    for(int j = 0; j < N + 1; j++)
      fp << now[0][j] << " ";
    fp << "\n";
  }
  for(int i = 1; i < size; i++) {
    for(int j = 0; j < N + 1; j++)
      fp << now[i][j] << " ";
    fp << "\n";
  }
  if(the_mpi.MyID == the_mpi.NProcs - 1){
    for(int j = 0; j < N + 1; j++)
      fp << now[size][j] << " ";
    fp << "\n";
  }
}


int main(int argc, char** argv)
{   
  // initialize MPI
  MPI_stuff the_mpi(argc, argv);

  MPI_Status status;
  MPI_Request req_send10, req_send20, req_recv10, req_recv20;

  /* breakup compute grid amoung processors and initialize*/
  int size = compute_my_size(the_mpi);
  Doub2D MeshA(boost::extents[size+1][N+1]);
  Doub2D MeshB(boost::extents[size+1][N+1]);
  // Set up raw pointers for fast swap of now and old arrays.  Normally one should use
  // shared pointers and a move for something like this but unfortuantely the boost multi_array
  // doesn't support move sematics and one is then left with a swap which would physically copy
  // every entry of the arrays which is way too slow.
  Doub2D *now = &MeshA;
  Doub2D *old = &MeshB;
  Doub2D *tmp;

  initialize(*old, *now, size, the_mpi);

  /* do one iteration and work out global error */
  double maxerrG;
  double maxerr = iteration(*old, *now, 1, size);
  MPI_Allreduce(&maxerr, &maxerrG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  /* Main loop */
  int iter=0;
  while(maxerrG > Tol){
    tmp = now;
    now = old;
    old = tmp;

    /* send/receive at top of compute block */
    req_send10 = req_recv20 = MPI_REQUEST_NULL;
    if(the_mpi.MyID < the_mpi.NProcs - 1){
      MPI_Isend(&(*old)[size-1][1], N-1, MPI_DOUBLE, the_mpi.MyID+1, 10,
		MPI_COMM_WORLD, &req_send10);
      MPI_Irecv(&(*old)[size][1], N-1, MPI_DOUBLE, the_mpi.MyID+1, 20,
		MPI_COMM_WORLD,&req_recv20);
    }
    /* send/receive at bottom of compute block */
    req_send20 = req_recv10 = MPI_REQUEST_NULL;
    if(the_mpi.MyID > 0){
      MPI_Isend(&(*old)[1][1], N-1, MPI_DOUBLE, the_mpi.MyID-1, 20,
		MPI_COMM_WORLD, &req_send20);
      MPI_Irecv(&(*old)[0][1], N-1, MPI_DOUBLE, the_mpi.MyID-1, 10,
		MPI_COMM_WORLD, &req_recv10);
    }
    /* update interior of compute block excluding beside boundaries */
    maxerr = iteration(*old, *now, 2, size-1);

    /* update compute block beside boundaries as available */
    /* top edge */
    if(the_mpi.MyID < the_mpi.NProcs - 1)
      MPI_Wait(&req_recv20, &status);
    maxerr = fmax(maxerr, iteration(*old, *now, size-1, size));

    /* bottom edge */
    if(the_mpi.MyID > 0)
      MPI_Wait(&req_recv10, &status);
    maxerr = fmax(maxerr, iteration(*old, *now, 1, 2));
  
    /* find global error */
    MPI_Allreduce(&maxerr, &maxerrG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    iter++;
  }
  if (the_mpi.MyID == 0) cout << "Iterations: " << iter << "\n";  

  /* Output result */
  output(*now, size, the_mpi);
  
  return 0;
}
