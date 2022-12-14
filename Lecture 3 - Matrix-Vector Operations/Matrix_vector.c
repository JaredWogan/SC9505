////////////////////////////////////////////
// Matrix-vector multiplication code Ab=c //
////////////////////////////////////////////

// Note that I will index arrays from 0 to n-1.
// Here workers do all the work and boss just handles collating results
// and sending info about A.

// include, definitions, globals etc here
#include <stdio.h>
#include <stdlib.h>
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


void GetArraySize(int *nrows, int *ncols, struct mpi_vars *the_mpi)
{
  if(the_mpi->MyID == 0){
    printf("Please enter the number of rows -> ");
    fflush(stdout); // Sometimes needed but shouldn't be
    scanf("%d",nrows);
    printf("Please enter the number of columnss -> ");
    fflush(stdout); // Sometimes needed but shouldn't be
    scanf("%d",ncols);
  }

  // send everyone nrows, ncols
  int buf[2];
  buf[0]= *nrows; buf[1]= *ncols;
  MPI_Bcast(buf,2,MPI_INT,0,MPI_COMM_WORLD);
  if (the_mpi->MyID != 0) {
    *nrows=buf[0]; *ncols=buf[1];
  }
}


void SetupArrays(int nrows, int ncols, double ***A, double *b, double **c, double  **Arow, struct mpi_vars *the_mpi)
{
  // Boss part
  if (the_mpi->MyID == 0) {
    // Allocate A
    *A = (double**) calloc(nrows,sizeof(double*));
    (*A)[0] = (double *) calloc(nrows*ncols, sizeof(double));
    for (int i=1; i<nrows; i++) {
      (*A)[i] = (*A)[i-1]+ ncols;
    }

    // Initialize A
    for (int i=0; i<nrows; ++i)
      for (int j=0; j<ncols; ++j) {
	if (i==j) (*A)[i][j] = 1.0;
	else (*A)[i][j] = 0.0;
      }

    // initialize b 
    for (int i=0; i<ncols; ++i)
      b[i] = 1.0;

    // Allocate space for c, the answer
    *c = (double *) calloc(ncols, sizeof(double));
  }
  // Worker part
  else {
    // Allocate space for 1 row of A
    *Arow = (double *) calloc(ncols, sizeof(double));
  }

  // send b to every worker process, note b is an array and b=&b[0] 
  MPI_Bcast(b,ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


void Output(double *c, int nrows, struct mpi_vars *the_mpi)
{
 if (the_mpi->MyID == 0) {
   printf("( %f",c[0]);
   for (int i=1; i<nrows; ++i)
     printf(",%f ",c[i]);
   printf(")\n");
 }
}

int main(int argc, char** argv)
{// initialize MPI
  struct mpi_vars the_mpi = mpi_start(argc, argv);

  // determine/distribute size of arrays here
  int nrows=0, ncols=0;
  GetArraySize(&nrows,&ncols,&the_mpi);
  //printf("processor %i has %i X %i \n",the_mpi.MyID,nrows,ncols);

  // assume A will have rows 0,nrows-1 and columns 0,ncols-1, so b is 0,ncols-1
  // so c must be 0,nrows-1.  Note declarations need to be out of if block to
  // avoid going out of scope.
  double** A = NULL; // Only Boss will use this one
  double *b = (double *) calloc(ncols, sizeof(double));
  double *c = NULL;     // Only boss uses this one
  double * Arow = NULL; // Only workers will use this one
  SetupArrays(nrows, ncols, &A, b, &c, &Arow, &the_mpi); // also sends b to everyone

  MPI_Status status;
 // Boss part
 if (the_mpi.MyID == 0) {
   // send one row to each worker tagged with row number, assume nprocs<nrows
   int rowsent=0;
   for (int i=1; i< the_mpi.NProcs; i++) {
     // Note A is a 2D array so A[rowsent]=&A[rowsent][0]
     MPI_Send(A[rowsent], ncols, MPI_DOUBLE,i,rowsent+1,MPI_COMM_WORLD);
     //printf("Boss sent row %i \n",rowsent);
     rowsent++;
   }

   for (int i=0; i<nrows; i++) {
     double ans;
     MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
     int sender = status.MPI_SOURCE;
     int anstype = status.MPI_TAG;            //row number+1
     c[anstype-1] = ans;
     if (rowsent < nrows) {                // send new row
       MPI_Send(A[rowsent],ncols,MPI_DOUBLE,sender,rowsent+1,MPI_COMM_WORLD);
       //printf("Boss sent row %i \n",rowsent);
       rowsent++;
     }
     else {       // tell sender no more work to do via a 0 TAG
       //printf("Boss sends kill signal\n");
       //fflush(stdout);
       MPI_Send(MPI_BOTTOM,0,MPI_DOUBLE,sender,0,MPI_COMM_WORLD);
     }
   }
 }
 // Worker part: compute dot products of Arow.b until done message recieved
 else {
   // Get a row of A
   MPI_Recv(Arow,ncols,MPI_DOUBLE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
   int crow; // which row did we get
   while(status.MPI_TAG != 0) {
     crow = status.MPI_TAG;
     //printf("Worker %i got row %i\n",the_mpi.MyID,crow-1);

     // work out Arow.b
     double ans=0.0;
     for (int i=0; i< ncols; i++)
       ans+=Arow[i]*b[i];
     
     // Send answer of Arow.b back to boss and get another row to work on
     MPI_Send(&ans,1,MPI_DOUBLE, 0, crow, MPI_COMM_WORLD);
     //printf("Worker %i sent c row %i\n",the_mpi.MyID,crow-1);
     MPI_Recv(Arow,ncols,MPI_DOUBLE, 0, MPI_ANY_TAG,MPI_COMM_WORLD,&status); 
   }
   //printf("Worker %i got kill tag\n",the_mpi.MyID);
   //fflush(stdout);
 }

 // output c here on Boss node
 Output(c,nrows,&the_mpi);

 //free any allocated space here
 if (the_mpi.MyID == 0) {
   free(A[0]);
   free(A);
   free(c);
 }
 else
   free(Arow);
 free(b);

 MPI_Finalize();
}



