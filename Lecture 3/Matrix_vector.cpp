////////////////////////////////////////////
// Matrix-vector multiplication code Ab=c //
////////////////////////////////////////////

// Note that I will index arrays from 0 to n-1.
// Here workers do all the work and boss just handles collating results
// and sending infor about A.

// include, definitions, globals etc here
#include <iostream>
#include "boost/multi_array.hpp"
#include "mpi.h"

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


void GetArraySize(int &nrows, int &ncols, MPI_stuff &the_mpi)
{
  if(the_mpi.MyID == 0){
    std::cout << "Please enter the number of rows -> ";
    std::cin >> nrows;
    std::cout << "Please enter the number of columns -> ";
    std::cin >> ncols;
  }

  // send everyone nrows, ncols
  int buf[2];
  buf[0]=nrows; buf[1]=ncols;
  MPI_Bcast(buf,2,MPI_INT,0,MPI_COMM_WORLD);
  if (the_mpi.MyID != 0) {
    nrows=buf[0]; ncols=buf[1];
  }
}

void SetupArrays(int nrows, int ncols, boost::multi_array<double, 2> &A, vector<double> &b, vector<double> &c, vector<double> &Arow, MPI_stuff &the_mpi)
{
  // Boss part
  if (the_mpi.MyID == 0) {
    // Set size of A
    A.resize(boost::extents[nrows][ncols]);

    // Initialize A to identity
    for (int i=0; i<nrows; ++i)
      for (int j=0; j<ncols; ++j) {
	if (i==j) A[i][j] = 1.0;
	else A[i][j] = 0.0;
      }

    // Initialize b 
    for (int i=0; i<ncols; ++i)
      b[i] = 1.0;

    // Allocate space for c, the answer
    c.reserve(nrows); c.resize(nrows);
  }
  // Worker part
  else {
    // Allocate space for 1 row of A
    Arow.reserve(ncols); Arow.resize(ncols);
  }
  
  // send b to every worker process, note b is a std::vector so b and &b[0] not same 
  MPI_Bcast(&b[0],ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Output(vector<double> &c, MPI_stuff &the_mpi)
{
 if (the_mpi.MyID == 0) {
   std::cout << "( " << c[0];
   for (int i=1; i< c.size(); ++i)
     std::cout <<  ", " << c[i];
   std::cout << ")\n";
 }
}

int main(int argc, char** argv)
{
  // initialize MPI
  MPI_stuff the_mpi(argc, argv);

  // determine/distribute size of arrays here
  int nrows=0, ncols=0;
  GetArraySize(nrows,ncols,the_mpi);
  //cout << "processor " << the_mpi.MyID << " has " << nrows << " X " << ncols << " \n";

  // assume A will have rows 0,nrows-1 and columns 0,ncols-1, so b is 0,ncols-1
  // so c must be 0,nrows-1.  Note declarations need to be outside of if block to
  // avoid going out of scope.
  boost::multi_array<double, 2> A; // Only Boss will use this one so leave sizeless for now
  std::vector<double> b(ncols);
  std::vector<double> c;   // Only Boss uses so leave sizeless for now
  std::vector<double> Arow;// Only workers use so leave sizeless for now
  SetupArrays(nrows, ncols, A, b, c, Arow, the_mpi);  // also sends b to everyone

  MPI_Status status;
 // Boss part
 if (the_mpi.MyID == 0) {
   // send one row to each worker tagged with row number, assume nprocs<nrows
   int rowsent=0;
   for (int i=1; i< the_mpi.NProcs; i++) {
     MPI_Send(&A[rowsent][0], ncols, MPI_DOUBLE,i,rowsent+1,MPI_COMM_WORLD);
     //cout << "Boss sent row " << rowsent << "\n";
     rowsent++;
   }

   for (int i=0; i<nrows; i++) {
     double ans;
     MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
     int sender = status.MPI_SOURCE;
     int anstype = status.MPI_TAG;            //row number+1
     c[anstype-1] = ans;
     if (rowsent < nrows) {                // send new row
       MPI_Send(&A[rowsent][0],ncols,MPI_DOUBLE,sender,rowsent+1,MPI_COMM_WORLD);
       //cout << "Boss sent row " << rowsent << "\n";
       rowsent++;
     }
     else {       // tell sender no more work to do via a 0 TAG
       //cout << "Boss sends kill signal\n";
       MPI_Send(MPI_BOTTOM,0,MPI_DOUBLE,sender,0,MPI_COMM_WORLD);
     }
   }
 }
 // Worker part: compute dot products of Arow.b until done message recieved
 else {
   // Get a row of A
   MPI_Recv(&Arow[0],ncols,MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
   int crow; // which row did we get
   while(status.MPI_TAG != 0) {
     crow = status.MPI_TAG;
     //cout << "Worker " << the_mpi.MyID << " got row " << crow-1 << "\n";

     // work out Arow.b
     double ans=0.0;
     for (int i=0; i< ncols; i++)
       ans+=Arow[i]*b[i];
     
     // Send answer of Arow.b back to boss and get another row to work on
     MPI_Send(&ans,1,MPI_DOUBLE, 0, crow, MPI_COMM_WORLD);
     //cout<< "Worker "<< the_mpi.MyID << " sent c row "<< crow-1 << ", "<< ans << "\n";
     MPI_Recv(&Arow[0],ncols,MPI_DOUBLE, 0, MPI_ANY_TAG,MPI_COMM_WORLD,&status); 
   }
   //cout << "Worker " << the_mpi.MyID << " got kill tag\n";
 }

 // output c here on Boss node
 Output(c, the_mpi);
}



