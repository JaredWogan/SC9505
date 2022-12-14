#include <stdio.h>

// compile with
// gcc Blas_dgemm.cpp -lblas

// dgemm_ is a symbol in the BLAS library files
extern int dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);

int main(void)
{
    // Method I
    // store as normal C-style array in row-major form
    // a is 2x3, b is 3x2, c is 2x2 and storage is row-major here
    double A2[2][3];
    double B2[3][2];
    double C2[2][2];

    A2[0][0] = 0.11;
    A2[0][1] = 0.12;
    A2[0][2] = 0.13;
    A2[1][0] = 0.21;
    A2[1][1] = 0.22;
    A2[1][2] = 0.23;
    B2[0][0] = 1011;
    B2[0][1] = 1012;
    B2[1][0] = 1021;
    B2[1][1] = 1022;
    B2[2][0] = 1031;
    B2[2][1] = 1032;
    C2[0][0] = 0.00;
    C2[0][1] = 0.00;
    C2[1][0] = 0.00;
    C2[1][1] = 0.00;

    // We want A.B but we have A and B in row-major so need to "transpose" to
    //  get in col-major form
    char transa = 'T', transb = 'T'; // op(A) = A, similar for B
    double alpha = 1.0, beta = 0.0;
    int m = 2, n = 2, k = 3;       // m=rows of op(A), n= cols of op(B), k= cols of op(A)
                                   // and rows of op(B)
    int lda = 3, ldb = 2, ldc = 2; // leading dimensions of A, B, C as declared is
                                   // fastest changing index, ie. last index in C-style

    /* Compute C = alpha*op(A)*op(B) + beta*C  using DGEEM and C is overwritten */
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, &A2[0][0], &lda,
           &B2[0][0], &ldb, &beta, &C2[0][0], &ldc);

    // Note that we get c back in col-major order so output transpose
    printf("[ %g, %g\n", C2[0][0], C2[1][0]);
    printf(" %g, %g ]\n", C2[0][1], C2[1][1]);

    /************************************************************************/
    // Method II
    // store as 2D array in col-major form
    // a is 2x3, b is 3x2, c is 2x2 and storage is row-major here but
    // let's interpret first index as column and 2nd as row
    double A3[3][2];
    double B3[2][3];
    double C3[2][2];

    A3[0][0] = 0.11;
    A3[1][0] = 0.12;
    A3[2][0] = 0.13;
    A3[0][1] = 0.21;
    A3[1][1] = 0.22;
    A3[2][1] = 0.23;
    B3[0][0] = 1011;
    B3[1][0] = 1012;
    B3[0][1] = 1021;
    B3[1][1] = 1022;
    B3[0][2] = 1031;
    B3[1][2] = 1032;
    C3[0][0] = 0.00;
    C3[1][0] = 0.00;
    C3[0][1] = 0.00;
    C3[1][1] = 0.00;

    // We want A.B but we have A and B in col-major so no "transpose" needed
    transa = 'N';
    transb = 'N';
    alpha = 1.0;
    beta = 0.0;
    m = 2;
    n = 2;
    k = 3; // m=rows of op(A), n= cols of op(B), k= cols of op(A)
           // and rows of op(B)
    lda = 2;
    ldb = 3;
    ldc = 2; // leading dimensions of A, B, C as declared where
             // "leading" means fastest changing which is actually the last in C

    /* Compute C = alpha*op(A)*op(B) + beta*C  using DGEEM and C is overwritten */
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, &A3[0][0], &lda,
           &B3[0][0], &ldb, &beta, &C3[0][0], &ldc);

    // Note that we get c back in col-major order so output with row-column switch
    printf("[ %g, %g\n", C3[0][0], C3[1][0]);
    printf(" %g, %g ]\n", C3[0][1], C3[1][1]);

    /************************************************************************/
    // Method III
    //  store as linear array in row-major form
    //  a is 2x3, b is 3x2, c is 2x2 and storage is row-major here
    //  so in col-major order we have a as 3x2, b as 2x3, and c as 2x2
    double a[] = {0.11, 0.12, 0.13, 0.21, 0.22, 0.23};
    double b[] = {1011, 1012, 1021, 1022, 1031, 1032};
    double c[] = {0.00, 0.00, 0.00, 0.00};

    // We want A.B but need to "transpose" to col-major order (not real transpose)
    //  to get actual A and actual B
    transa = 'T';
    transb = 'T'; // op(A) = A transpose, similar for B
    alpha = 1.0;
    beta = 0.0;
    m = 2;
    n = 2;
    k = 3; // m=rows of op(A), n= cols of op(B), k= cols of op(A)
           // and rows of op(B)
    lda = 3;
    ldb = 2;
    ldc = 2; // leading dimensions of A, B, C as declared

    /* Compute C = alpha*op(A)*op(B) + beta*C  using DGEEM and C is overwritten */
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, &a[0], &lda,
           &b[0], &ldb, &beta, &c[0], &ldc);

    // Note that we get the transpose of c back again due to col-major order
    printf("[ %g, %g\n", c[0], c[2]);
    printf(" %g, %g ]\n", c[1], c[3]);

    /*************************************************************************/
    // Method IV
    // store as linear array in col-major form
    // a is 2x3, b is 3x2, c is 2x2 and storage is col-major here
    double A[] = {0.11, 0.21, 0.12, 0.22, 0.13, 0.23};
    double B[] = {1011, 1021, 1031, 1012, 1022, 1032};
    double C[] = {0.00, 0.00, 0.00, 0.00};

    // We want A.B which are already in col-major form so don't transpose
    transa = 'N';
    transb = 'N'; // op(A) = A, similar for B
    alpha = 1.0;
    beta = 0.0;
    m = 2;
    n = 2;
    k = 3; // m=rows of op(A), n= cols of op(B), k= cols of op(A)
           // and rows of op(B)
    lda = 2;
    ldb = 3;
    ldc = 2; // leading dimensions of A, B, C as declared

    /* Compute C = alpha*op(A)*op(B) + beta*C  using DGEEM and C is overwritten */
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, &A[0], &lda,
           &B[0], &ldb, &beta, &C[0], &ldc);

    // Note that we get c back in col-major order
    printf("[ %g, %g\n", C[0], C[2]);
    printf(" %g, %g ]\n", C[1], C[3]);

    return 0;
}
