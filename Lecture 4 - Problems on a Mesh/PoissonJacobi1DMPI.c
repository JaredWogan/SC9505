#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
// Solve Poisson's equations using the Jacobi method, with MPI parallization
// dividing the grid into a 1D array of processors 
// Compile with mpicc PoissonJacobi1DMPI.c -lm

#define N  1000
#define Tol  0.0001

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


// breakup array size N in 1D into parts for each processor
int compute_my_size(struct mpi_vars the_mpi)
{
  int remainder = (N - 1) % the_mpi.NProcs;
  int size = (N - 1 - remainder)/the_mpi.NProcs;
  if(the_mpi.MyID < remainder)  // extra rows added for MyID < remainder 
    size = size + 2;
  else
    size = size + 1;
  return size;
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

void initialize(double **old, double **new, int size, struct mpi_vars the_mpi)
{ /* Assume interior is already initialized, just need to set boundaries */
  /* B.Cs set to a constant, 1 here */

  for(int i = 0; i < size + 1; i++)
    new[i][0] = new[i][N] = old[i][0] = old[i][N] = 1;
  if(the_mpi.MyID == 0)
    for(int j = 1; j < N; j++)
      new[0][j] = old[0][j] = 1;
  if(the_mpi.MyID == the_mpi.NProcs - 1)
    for(int j = 1; j < N; j++)
      new[size][j] = old[size][j] = 1;
}

double iteration(double **old, double **new, int start, int finish)
{   /* Jacobi iteration */
    double maxerr = 0;
    for(int i = start; i < finish; i++)
       for(int j = 1; j < N; j++){
          new[i][j] = 0.25*(old[i+1][j] + old[i-1][j] +
                            old[i][j+1] + old[i][j-1]);

	  maxerr = fmax(maxerr, fabs(new[i][j] - old[i][j]));
       }
    return (maxerr);
}


/* Output result */
void output(double **new, int size, struct mpi_vars the_mpi)
{
  char str[20];
  FILE *fp;

  sprintf(str,"Solution%d.Txt",the_mpi.MyID);
  fp = fopen(str,"w");
  if(the_mpi.MyID == 0) {
    for(int j = 0; j < N + 1; j++)
      fprintf(fp,"%6.4f ",new[0][j]);
    fprintf(fp,"\n");
  }
  for(int i = 1; i < size; i++) {
    for(int j = 0; j < N + 1; j++)
      fprintf(fp,"%6.4f ",new[i][j]);
    fprintf(fp,"\n");
  }
  if(the_mpi.MyID == the_mpi.NProcs - 1){
    for(int j = 0; j < N + 1; j++)
      fprintf(fp,"%6.4f ",new[size][j]);
    fprintf(fp,"\n");
  }
  fclose(fp);
}


int main(int argc, char** argv)
{   
  struct mpi_vars the_mpi = mpi_start(argc, argv);

  MPI_Status status;
  MPI_Request req_send10, req_send20, req_recv10, req_recv20;

  /* breakup compute grid amoung processors and initialize*/
  int size = compute_my_size(the_mpi);
  double **new = matrix(size+1, N+1);
  double **old = matrix(size+1, N+1);
  initialize(old, new, size, the_mpi);
  double **tmp; //used for swapping old and new arrays
  
  /* do one iteration and work out global error */
  double maxerrG;
  double maxerr = iteration(old, new, 1, size);
  MPI_Allreduce(&maxerr, &maxerrG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  /* Main loop */
  int iter=0;
  double tswap=0, titer=0, tinit, tfinal;
  while(maxerrG > Tol){
    tinit = MPI_Wtime();
    tmp = new;
    new = old;
    old = tmp;
    tfinal = MPI_Wtime();
    tswap += tfinal-tinit;

    /* send/receive at top of compute block */
    req_send10 = req_recv20 = MPI_REQUEST_NULL;
    if(the_mpi.MyID < the_mpi.NProcs - 1){
      MPI_Isend(&old[size-1][1], N-1, MPI_DOUBLE, the_mpi.MyID+1, 10,
		MPI_COMM_WORLD, &req_send10);
      MPI_Irecv(&old[size][1], N-1, MPI_DOUBLE, the_mpi.MyID+1, 20,
		MPI_COMM_WORLD,&req_recv20);
    }
    /* send/receive at bottom of compute block */
    req_send20 = req_recv10 = MPI_REQUEST_NULL;
    if(the_mpi.MyID > 0){
      MPI_Isend(&old[1][1], N-1, MPI_DOUBLE, the_mpi.MyID-1, 20,
		MPI_COMM_WORLD, &req_send20);
      MPI_Irecv(&old[0][1], N-1, MPI_DOUBLE, the_mpi.MyID-1, 10,
		MPI_COMM_WORLD, &req_recv10);
    }
    /* update interior of compute block excluding beside boundaries */
    tinit = MPI_Wtime();
    maxerr = iteration(old, new, 2, size-1);
    tfinal = MPI_Wtime();
    titer += tfinal-tinit;

    /* update compute block beside boundaries as available */
    /* top edge */
    if(the_mpi.MyID < the_mpi.NProcs - 1)
      MPI_Wait(&req_recv20, &status);
    maxerr = fmax(maxerr, iteration(old, new, size-1, size));

    /* bottom edge */
    if(the_mpi.MyID > 0)
      MPI_Wait(&req_recv10, &status);
    maxerr = fmax(maxerr, iteration(old, new, 1, 2));
  
    /* find global error */
    MPI_Allreduce(&maxerr, &maxerrG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    iter++;
  }
  if (the_mpi.MyID == 0) printf("Iterations: %d\n",iter);
  printf("tswap = %f titer = %f \n", tswap,titer);

  /* Output result */
  output(new, size, the_mpi);
  
  free(old[0]);
  free(old);
  free(new[0]);
  free(new);
  MPI_Finalize();
  return 0;
}
