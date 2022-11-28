////////////////////////////////////////////
// Matrix-vector multiplication code Ab=c //
////////////////////////////////////////////

// Note that I will index arrays from 0 to n-1.
// Here workers do all the work and boss just handles collating results
// and sending infor about A.

// include, definitions, globals etc here
#include <iostream>
#include <iomanip>
#include "boost/multi_array.hpp"
#include <mpi.h>
#include <random>

using namespace std;

void GetArraySize(int &output, int &nrows, int &ncols, int argc, char **argv)
{
    if (argc  == 4) {
        output = atoi(argv[1]);
        nrows = atoi(argv[2]);
        ncols = atoi(argv[3]);
    } else {
        cout << "Please enter the number of rows -> ";
        cin >> nrows;
        cout << "Please enter the number of columns -> ";
        cin >> ncols;
    }
}

void SetupArrays(int nrows, int ncols, boost::multi_array<double, 2> &A, vector<double> &b, vector<double> &c) {
    uniform_real_distribution<double> unif(-1.0, 10);
    default_random_engine re;
    // Set size of A
    A.resize(boost::extents[nrows][ncols]);

    // Initialize A to identity
    for (int i = 0; i < nrows; ++i)
        for (int j = 0; j < ncols; ++j)
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

    // Initialize b
    for (int i = 0; i < ncols; ++i)
    {
        // b[i] = 1.0;
        // b[i] = (double) i;
        b[i] = unif(re);
    }

    // Allocate space for c, the answer
    c.reserve(nrows);
    c.resize(nrows);
}

void Output(int output, vector<double> &c)
{
    if (output) {
        cout << "( " << c[0];
        for (int i = 1; i < c.size(); ++i)
            cout << ", " << c[i];
        cout << ")\n";
    }
}

int main(int argc, char **argv)
{
    // determine/distribute size of arrays here
    int output = 1, nrows = 0, ncols = 0;
    GetArraySize(output, nrows, ncols, argc, argv);

    // assume A will have rows 0,nrows-1 and columns 0,ncols-1, so b is 0,ncols-1
    // so c must be 0,nrows-1.  Note declarations need to be outside of if block to
    // avoid going out of scope.
    boost::multi_array<double, 2> A; // Only Boss will use this one so leave sizeless for now
    vector<double> b(ncols);
    vector<double> c;                             // Only Boss uses so leave sizeless for now
    SetupArrays(nrows, ncols, A, b, c); // also sends b to everyone

    // Timing variables
    double ans;
    double total_time = 0;

    total_time = MPI_Wtime();
    for (int i = 0; i < nrows; i++)
    {
        c[i] = 0;
        for (int j = 0; j < ncols; j++) {
            c[i] += A[i][j] * b[j];
        }
    }
    total_time = MPI_Wtime() - total_time;

    // output c here on Boss node
    Output(output, c);

    printf("Total time = %.10f\n", total_time);
}
