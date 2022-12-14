// Solve Poisson's equations using the Jacobi method, with MPI parallization
// dividing the grid into a 2D array of processors 
// Compile line:
// mpicc -O3 PoissonJacobi2DMPI.c -lm -o pois2d

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "mpi.h"
#define N  21             /* N x N is size of interior grid */
#define Tol  0.00001

int isperiodic[2]={0,0};  // Flags for periodic boundary conditions are off

// MPI Stuff
struct mpi_vars {
  int NProcs;
  int MyID;
  MPI_Comm D1Comm; // 1D communicator
  int nbrdown, nbrup;

  MPI_Comm D2Comm; // 2D communicator
  int dims[2], IDcoord[2]; //  processors in each direction in 2D
  int nbrleft, nbrright;
}; 

struct mpi_vars mpi_start(int argc, char** argv, int dim)
{
  struct mpi_vars mpi;

  MPI_Init(&argc, &argv);

  // We always want a 1D array of processors so just duplicate default 
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi.D1Comm);

  MPI_Comm_size(mpi.D1Comm, &mpi.NProcs);
  if (dim == 1) {
    MPI_Comm_rank(mpi.D1Comm, &mpi.MyID);
    mpi.nbrup = mpi.MyID + 1;
    mpi.nbrdown = mpi.MyID -1;

    mpi.D2Comm = MPI_COMM_NULL;
  }

  // We sometimes want a 2D array so define if requested
  if (dim == 2) {
    // Find an integer closest to the square root of the processor number
    mpi.dims[0] = (int) (sqrt(((double)mpi.NProcs))+2.*FLT_EPSILON);
    // Assign as many as possible to other dimension (could leave some unused)
    mpi.dims[1] = mpi.NProcs/mpi.dims[0];

    /* Create cartesian grid of processors */
    MPI_Cart_create(MPI_COMM_WORLD, 2, mpi.dims, isperiodic, 1, &mpi.D2Comm);
    MPI_Comm_rank(mpi.D2Comm, &mpi.MyID); //need rank from 2D Comm
    if (mpi.MyID == 0)
      printf("number of processors %i: %i x %i\n", 
	     mpi.NProcs, mpi.dims[0], mpi.dims[1]);
  
    MPI_Cart_coords(mpi.D2Comm, mpi.MyID, 2, mpi.IDcoord);
    MPI_Cart_shift(mpi.D2Comm, 0, 1, &mpi.nbrleft, &mpi.nbrright);
    MPI_Cart_shift(mpi.D2Comm, 1, 1, &mpi.nbrdown, &mpi.nbrup);
  }

  return mpi;
}


// breakup array size N x N in 2D into parts for each processor
void compute_my_size(int *xsize, int *ysize, int *sameview, struct mpi_vars mpi)
{
  int xremainder = N  % mpi.dims[0];
  *xsize = (N - xremainder)/mpi.dims[0];
  if(mpi.IDcoord[0] < xremainder)
    *xsize = *xsize + 2;
  else 
    *xsize = *xsize + 1;
  
  int yremainder = N  % mpi.dims[1];
  *ysize = (N - yremainder)/mpi.dims[1];
  if(mpi.IDcoord[1] < yremainder)
    *ysize = *ysize + 2;
  else
    *ysize = *ysize + 1;

  *sameview = !(xremainder || yremainder);

  printf("compute grid (1 + %i + 1) x (1 + %i + 1) set on processor %i \n",
	*xsize-1,*ysize-1,mpi.MyID);
}


double **matrix(int m, int n)
{
    /* Note that you must allocate the array as one block in order to use */
    /* MPI derived data types on them.  Note also that calloc initializes */
    /* all entries to zero. */
    
    double ** ptr = (double **)calloc(m, sizeof(double *));
    ptr[0]=(double *)calloc(m*n, sizeof(double));
    for(int i = 1; i < m ;i++)
      ptr[i]=ptr[i-1]+n;
    return (ptr);
}


void initialize(double **old, double **new, int xsize, int ysize, struct mpi_vars mpi)
{ /* Assume interior is already initialized, just need to set boundaries */
  /* B.C.s set to a constant, 1 here */
  /* left edge */
  if (mpi.IDcoord[0] == 0)
    for (int i=0; i< ysize + 1; i++)
      new[i][0]=old[i][0]=1.0;
  /* right edge */
  if (mpi.IDcoord[0] == mpi.dims[0]-1)
    for (int i=0; i< ysize + 1; i++)
      new[i][xsize]=old[i][xsize]=1.0;
  /* bottom edge */
  if(mpi.IDcoord[1] == 0)
    for(int j = 1; j < xsize+1; j++)
      new[0][j] = old[0][j] = 1.0;
  /* top edge */
  if(mpi.IDcoord[1] == mpi.dims[1] - 1)
    for(int j = 1; j < xsize+1; j++)
      new[ysize][j] = old[ysize][j] = 1.0;
  
  printf("boundary conditions set on processor %i\n",mpi.MyID);
}


double iteration(double **old, double **new, int xstart, int xfinish, 
		int ystart, int yfinish)
{  /* Jacobi iteration */ 

  double maxerr = 0;
  for(int i = ystart; i < yfinish; i++)
    for(int j = xstart; j < xfinish; j++){
      new[i][j] = 0.25*(old[i+1][j] + old[i-1][j] +
			old[i][j+1] + old[i][j-1]);
      maxerr = fmax(maxerr,fabs(new[i][j] - old[i][j]));
    }
  return (maxerr);
}


/* Output result */
void output(double **new, int xsize, int ysize, struct mpi_vars mpi)
{
  char str[20];
  FILE *fp;
  sprintf(str,"result%d_%d.txt",mpi.IDcoord[0],mpi.IDcoord[1]);
  fp = fopen(str,"wt");
  if(mpi.IDcoord[1] == 0) {
    if (mpi.IDcoord[0] == 0)
      fprintf(fp,"%6.4f ",new[0][0]);
    for(int j = 1; j < xsize; j++)
      fprintf(fp,"%6.4f ",new[0][j]);
    if (mpi.IDcoord[0] == mpi.dims[0]-1)
      fprintf(fp,"%6.4f ",new[0][xsize]);
    fprintf(fp,"\n");
  }
  for(int i = 1; i < ysize; i++) {
    if (mpi.IDcoord[0] == 0)
      fprintf(fp,"%6.4f ",new[i][0]);
    for(int j = 1; j < xsize; j++)
      fprintf(fp,"%6.4f ",new[i][j]);
    if (mpi.IDcoord[0] == mpi.dims[0]-1)
      fprintf(fp,"%6.4f ",new[i][xsize]);
    fprintf(fp,"\n");
  }
  if(mpi.IDcoord[1] == mpi.dims[1] - 1) {
    if (mpi.IDcoord[0] == 0)
      fprintf(fp,"%6.4f ",new[ysize][0]);
    for(int j = 1; j < xsize; j++)
      fprintf(fp,"%6.4f ",new[ysize][j]);
    if (mpi.IDcoord[0] == mpi.dims[0]-1)
      fprintf(fp,"%6.4f ",new[ysize][xsize]);
    fprintf(fp,"\n");
  }
  fclose(fp);
}

void output_onefile(double **new, int xsize, int ysize, struct mpi_vars mpi)
{
  /* Create derived datatype for local interior grid (output grid) */
  MPI_Datatype grid;
  int start[2] = {1, 1};  // indices of interior "origin"
  int arrsize[2] = {xsize+1, ysize+1};  // full local array size
  int gridsize[2] = {xsize+1 - 2, ysize+1 - 2};  // size of interior

  MPI_Type_create_subarray(2, arrsize, gridsize,
			   start, MPI_ORDER_FORTRAN, MPI_DOUBLE, &grid);
  MPI_Type_commit(&grid);

  /* Create derived type for file view, how local interior fits into global system */
  MPI_Datatype view;
  int nnx = xsize+1-2, nny = ysize+1-2; 
  int startV[2] = { mpi.IDcoord[0]*nnx, mpi.IDcoord[1]*nny };
  int arrsizeV[2] = { mpi.dims[0]*nnx, mpi.dims[1]*nny };
  int gridsizeV[2] = { nnx, nny };
  
  MPI_Type_create_subarray(2, arrsizeV, gridsizeV,
			   startV, MPI_ORDER_FORTRAN, MPI_DOUBLE, &view);
  MPI_Type_commit(&view);

  /* MPI IO */
  MPI_File fp;

  MPI_File_open(mpi.D2Comm, "output.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY,
		MPI_INFO_NULL, &fp);

  MPI_File_set_view(fp, 0, MPI_DOUBLE, view, "native", MPI_INFO_NULL);
  MPI_File_write_all(fp, &new[0][0], 1, grid, MPI_STATUS_IGNORE);
  MPI_File_close(&fp);
}


int main(int argc, char** argv)
{   
  // Start MPI with a 2D cartesian grid of processors
  struct mpi_vars mpi = mpi_start(argc, argv, 2);

  MPI_Status status;
  MPI_Request req_send10, req_send20, req_recv10, req_recv20;
  MPI_Request req_send30, req_send40, req_recv30, req_recv40;

  int maxit=200; 

  /* breakup compute grid amoung processors */
  int xsize,ysize;
  int sameview;
  compute_my_size(&xsize, &ysize, &sameview, mpi);
  double **new = matrix(ysize+1,xsize+1);
  double **old = matrix(ysize+1,xsize+1);
  double **tmp; //used for swapping old and new arrays
  initialize(old, new, xsize, ysize, mpi);

  /* Create data type to send columns of data */
  /* If you didn't get this to work, look at the comment in the matrix */
  /* allocation routine */
  MPI_Datatype stridetype;
  MPI_Type_vector(ysize-1,1,xsize+1,MPI_DOUBLE,&stridetype);
  MPI_Type_commit(&stridetype);
   
  /* do one iteration and work out global error */
  double maxerrG;
  double maxerr = iteration(old,new,1,xsize,1,ysize);
  MPI_Allreduce(&maxerr,&maxerrG,1,MPI_DOUBLE,MPI_MAX,mpi.D2Comm);
  if (mpi.MyID == 0) printf("initial error %f\n",maxerrG);
  
  /* Main loop */
  int iter=0;
  while(maxerrG > Tol && iter < maxit){
    tmp = new;
    new = old;
    old = tmp;

    /* send/receive at top of compute block */
    req_send10 = req_recv20 = MPI_REQUEST_NULL;
    if(mpi.IDcoord[1] < mpi.dims[1]-1){
      MPI_Isend(&old[ysize-1][1],xsize-1,MPI_DOUBLE,mpi.nbrup,10,mpi.D2Comm,&req_send10);
      MPI_Irecv(&old[ysize][1],xsize-1,MPI_DOUBLE,mpi.nbrup,20,mpi.D2Comm,&req_recv20);
    }
    /* send/receive at bottom of compute block */
    req_send20 = req_recv10 = MPI_REQUEST_NULL;
    if(mpi.IDcoord[1] > 0){
      MPI_Isend(&old[1][1],xsize-1,MPI_DOUBLE,mpi.nbrdown,20,mpi.D2Comm,&req_send20);
      MPI_Irecv(&old[0][1],xsize-1,MPI_DOUBLE,mpi.nbrdown,10,mpi.D2Comm,&req_recv10);
    }
    /* send/receive at right of compute block */
    req_send30 = req_recv40 = MPI_REQUEST_NULL;
    if(mpi.IDcoord[0] < mpi.dims[0]-1){
      MPI_Isend(&old[1][xsize-1],1,stridetype,mpi.nbrright,30,mpi.D2Comm,&req_send30);
      MPI_Irecv(&old[1][xsize],1,stridetype,mpi.nbrright,40,mpi.D2Comm,&req_recv40);
    }
    /* send/receive at left of compute block */
    req_send40 = req_recv30 = MPI_REQUEST_NULL;
    if(mpi.IDcoord[0] > 0){
      MPI_Isend(&old[1][1],1,stridetype,mpi.nbrleft,40,mpi.D2Comm,&req_send40);
      MPI_Irecv(&old[1][0],1,stridetype,mpi.nbrleft,30,mpi.D2Comm,&req_recv30);
    }
    /* update interior of compute block except beside boundaries */
    maxerr = iteration(old,new,2,xsize-1,2,ysize-1);

    /* fill in compute block beside boundaries as available */
    /* top edge */
    if(mpi.IDcoord[1] < mpi.dims[1]-1) MPI_Wait(&req_recv20,&status);
    maxerr = fmax(maxerr, iteration(old,new,2,xsize-1,ysize-1,ysize));

    /* bottom edge */
    if(mpi.IDcoord[1] > 0) MPI_Wait(&req_recv10,&status);
    maxerr = fmax(maxerr, iteration(old,new,2,xsize-1,1,2));

    /* right edge */
    if(mpi.IDcoord[0] < mpi.dims[0]-1) MPI_Wait(&req_recv40,&status);
    maxerr = fmax(maxerr, iteration(old,new,xsize-1,xsize,1,ysize));

    /* left edge */
    if(mpi.IDcoord[0] > 0) MPI_Wait(&req_recv30,&status);
    maxerr = fmax(maxerr, iteration(old,new,1,2,1,ysize));

    /* find global error */
    MPI_Allreduce(&maxerr,&maxerrG,1,MPI_DOUBLE,MPI_MAX,mpi.D2Comm);
    if (mpi.MyID == 0) printf(" error %f\r",maxerrG);

    iter++;
  }
  printf("\n");

  /* Output result */
  if (sameview)
    output_onefile(new, xsize, ysize, mpi);
  else
    output(new, xsize, ysize, mpi);

  free(old[0]);
  free(old);
  free(new[0]);
  free(new);
  MPI_Finalize();
  return 0;
}
