////////////////////////////////////////////
// Matrix-vector multiplication code Ab=c //
////////////////////////////////////////////

// Note that I will index arrays from 0 to n-1.
// Here workers do all the work and boss just handles collating results
// and sending infor about A.

// include, definitions, globals etc here
#include <iostream>
#include <iomanip>
#include <random>
#include "boost/multi_array.hpp"
#include "mpi.h"

extern "C"
{
    extern int dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
}

using namespace std;

void GetArraySize(
    int &output,
    int &nrows1,
    int &nrowscols12,
    int &ncols2,
    int argc,
    char **argv)
{
    if (argc == 5)
    {
        output = atoi(argv[1]);
        nrows1 = atoi(argv[2]);
        nrowscols12 = atoi(argv[3]);
        ncols2 = atoi(argv[4]);
    }
    else
    {
        cout << "Please enter the number of rows for the first matrix -> ";
        cin >> nrows1;
        cout << "Please enter the number of columns/rows for the first/second matrix-> ";
        cin >> nrowscols12;
        cout << "Please enter the number of columns for the second matrix-> ";
        cin >> ncols2;
    }

    // send everyone nrows, ncols
    int buf[3] = {nrows1, nrowscols12, ncols2};
    nrows1 = buf[0];
    nrowscols12 = buf[1];
    ncols2 = buf[2];
}

void SetupArrays(
    int nrows1,
    int nrowscols12,
    int ncols2,
    boost::multi_array<double, 2> &A,
    boost::multi_array<double, 2> &B,
    boost::multi_array<double, 2> &C
) {
    uniform_real_distribution<double> unif(-1.0, 10);
    default_random_engine re;

    // Set size
    A.resize(boost::extents[nrows1][nrowscols12]);
    B.resize(boost::extents[nrowscols12][ncols2]);
    C.resize(boost::extents[nrows1][ncols2]);

    // Initialize A
    for (int i = 0; i < nrows1; ++i)
    {
        for (int j = 0; j < nrowscols12; ++j)
        {
            // Identity
            // if (i == j)
            //     A[i][j] = 1.0;
            // else
            //     A[i][j] = 0.0;
            // Reverse Identity
            // if (i == (ncols - j - 1))
            //     A[i][j] = 1.0;
            // else
            //     A[i][j] = 0.0;
            // Random
            A[i][j] = unif(re);
        }
    }

    // Initialize B
    for (int i = 0; i < nrowscols12; ++i)
    {
        for (int j = 0; j < ncols2; ++j)
        {
            // Identity
            // if (i == j)
            //     B[i][j] = 1.0;
            // else
            //     B[i][j] = 0.0;
            // Reverse Identity
            // if (i == (ncols - j - 1))
            //     B[i][j] = 1.0;
            // else
            //     B[i][j] = 0.0;
            // Random
            B[i][j] = unif(re);
            // Other
            // if (i == j)
            //     B[i][j] = (double)i + 1;
            // else
            //     B[i][j] = 0;
        }
    }
}

void Output(int output, boost::multi_array<double, 2> &array)
{
    if (output)
    {
        cout << endl
             << fixed << setprecision(4);

        for (int i = 0; i < array.shape()[0]; i++)
        {
            for (int j = 0; j < array.shape()[1]; j++)
            {
                cout << array[i][j];
                if (j < array.shape()[1] - 1)
                    cout << ", ";
            }
            cout << endl;
        }
        cout << endl;

        cout << scientific;
    }
}

int main(int argc, char **argv)
{
    // determine/distribute size of arrays here
    int output = 1, nrows1 = 0, nrowscols12 = 0, ncols2 = 0;
    GetArraySize(output, nrows1, nrowscols12, ncols2, argc, argv);

    boost::multi_array<double, 2> A;
    boost::multi_array<double, 2> B;
    boost::multi_array<double, 2> C(boost::extents[0][0], boost::fortran_storage_order());
    SetupArrays(nrows1, nrowscols12, ncols2, A, B, C); // also sends b to everyone

    // Timing variables
    double total_time;

    total_time = MPI_Wtime();

    /* Compute C = alpha * op(A) * op(B) + beta * C using DGEEM and C is overwritten */
    dgemm_(
        new char('T'), // op(A) = A**T
        new char('T'), // op(B) = B**T
        &nrows1, // Specifies the number of rows of the matrix op(A)
        &ncols2, // Specifies the number of columns of the matrix op(B)
        &nrowscols12, // Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B)
        new double(1), // alpha
        &A[0][0], // A
        &nrowscols12, // LDA - Dimension of leading (fastest changing) index
        &B[0][0], // B
        &ncols2, // LDB - Dimension of leading (fastest changing) index
        new double(0), // beta
        &C[0][0], // C
        &nrows1 // LDC - Dimension of leading (fastest changing) index
    );

    total_time = MPI_Wtime() - total_time;
    printf("Total time = %.10f\n", total_time);

    // output c here on Boss node
    Output(output, C);
}
