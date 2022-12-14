#include <stdio.h>
#include "boost/multi_array.hpp"

// compile with
// g++ Blas_dgemm.cpp -lblas

// dgemm_ is a symbol in the BLAS library files
extern "C" {
  extern int dgemm_(char*,char*,int*,int*,int*,double*,double*,int*,double*, int*,double*,  double*, int*);
}

int main (void)
{
  // Method I
  // store as normal C-style array in row-major form
  // a is 2x3, b is 3x2, c is 2x2 and storage is row-major here
  boost::multi_array<double,2> A2(boost::extents[2][3]);
  boost::multi_array<double,2> B2(boost::extents[3][2]);
  boost::multi_array<double,2> C2(boost::extents[2][2]);

  A2[0][0] = 0.11; A2[0][1] = 0.12; A2[0][2] = 0.13;
  A2[1][0] = 0.21; A2[1][1] = 0.22; A2[1][2] = 0.23;
  B2[0][0] = 1011; B2[0][1] = 1012;
  B2[1][0] = 1021; B2[1][1] = 1022;
  B2[2][0] = 1031; B2[2][1] = 1032;
  C2[0][0] = 0.00; C2[0][1] = 0.00;
  C2[1][0] = 0.00; C2[1][1] = 0.00;

  //We want A.B but we have A and B in row-major so need to "transpose" to 
  // get in col-major form
  char transa='T', transb='T';  // op(A) = A, similar for B
  double alpha=1.0, beta=0.0;
  int m=2, n=2, k=3;   // m=rows of op(A), n= cols of op(B), k= cols of op(A) 
                       // and rows of op(B)
  int lda=3, ldb=2, ldc=2; // leading dimensions of A, B, C as declared is
                           // fastest changing index, ie. last index in C-style

  /* Compute C = alpha*op(A)*op(B) + beta*C  using DGEEM and C is overwritten */
  dgemm_(&transa, &transb, &m, &n, &k, &alpha, &A2[0][0], &lda, 
	 &B2[0][0], &ldb, &beta, &C2[0][0], &ldc);

  // Note that we get c back in col-major order so output transpose
  printf ("[ %g, %g\n", C2[0][0], C2[1][0]);
  printf (" %g, %g ]\n", C2[0][1], C2[1][1]);

  /************************************************************************/
  // Method II
  // store as 2D array in col-major form
  // a is 2x3, b is 3x2, c is 2x2 and storage is col-major here
  boost::multi_array<double,2> A3(boost::extents[2][3],boost::fortran_storage_order());
  boost::multi_array<double,2> B3(boost::extents[3][2],boost::fortran_storage_order());
  boost::multi_array<double,2> C3(boost::extents[2][2],boost::fortran_storage_order());

  A3[0][0] = 0.11; A3[0][1] = 0.12; A3[0][2] = 0.13;
  A3[1][0] = 0.21; A3[1][1] = 0.22; A3[1][2] = 0.23;
  B3[0][0] = 1011; B3[0][1] = 1012;
  B3[1][0] = 1021; B3[1][1] = 1022;
  B3[2][0] = 1031; B3[2][1] = 1032;
  C3[0][0] = 0.00; C3[0][1] = 0.00;
  C3[1][0] = 0.00; C3[1][1] = 0.00;

  //We want A.B but we have A and B in col-major so no "transpose" needed
  transa='N'; transb='N'; 
  alpha=1.0; beta=0.0;
  m=2; n=2; k=3;   // m=rows of op(A), n= cols of op(B), k= cols of op(A) 
                   // and rows of op(B)
  lda=2; ldb=3; ldc=2;  // leading dimensions of A, B, C as declared where 
            // "leading" means fastest changing which is actually the last in C

  /* Compute C = alpha*op(A)*op(B) + beta*C  using DGEEM and C is overwritten */
  dgemm_(&transa, &transb, &m, &n, &k, &alpha, &A3[0][0], &lda, 
	 &B3[0][0], &ldb, &beta, &C3[0][0], &ldc);

  // Note that we get c back in col-major order but we have stored in fortran form
  // so no output transpose is necessary here
  printf ("[ %g, %g\n", C3[0][0], C3[0][1]);
  printf (" %g, %g ]\n", C3[1][0], C3[1][1]);

 return 0;
}
