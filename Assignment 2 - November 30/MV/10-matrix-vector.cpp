////////////////////////////////////////////
// Matrix-vector multiplication code Ab=c //
////////////////////////////////////////////

// Note that I will index arrays from 0 to n-1.
// Here workers do all the work and boss just handles collating results
// and sending info about A.

// include, definitions, globals etc here
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include "boost/multi_array.hpp"
#include "mpi.h"

using namespace std;

class MPI_Obj
{
public:
    int size;
    int rank;

    MPI_Obj(int &argc, char **&argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    ~MPI_Obj()
    {
        MPI_Finalize();
    }
};

void GetArraySize(int &output, int &nrows, int &ncols, MPI_Obj &the_mpi, int argc, char **argv)
{
    if (the_mpi.rank == 0)
    {
        if (argc == 4)
        {
            output = atoi(argv[1]);
            nrows = atoi(argv[2]);
            ncols = atoi(argv[3]);
        }
        else
        {
            cout << "Please enter the number of rows -> ";
            cin >> nrows;
            cout << "Please enter the number of columns -> ";
            cin >> ncols;
        }
    }

    // send everyone nrows, ncols
    int buf[2] = {nrows, ncols};
    MPI_Bcast(buf, 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (the_mpi.rank != 0)
    {
        nrows = buf[0];
        ncols = buf[1];
    }
}

void SetupArrays(int nrows, int ncols, boost::multi_array<double, 2> &A, vector<double> &b, vector<double> &c, vector<double> &Arow, MPI_Obj &the_mpi)
{
    uniform_real_distribution<double> unif(-1.0, 10);
    default_random_engine re;
    // Boss part
    if (the_mpi.rank == 0)
    {
        // Set size of A
        A.resize(boost::extents[nrows][ncols]);

        // Initialize A
        for (int i = 0; i < nrows; ++i)
            for (int j = 0; j < ncols; ++j)
            {
                // Identity
                if (i == j)
                    A[i][j] = 1.0;
                else
                    A[i][j] = 0.0;
                // Reverse Identity
                // if (i == (ncols - j - 1))
                //     A[i][j] = 1.0;
                // else
                //     A[i][j] = 0.0;
                // Random
                // A[i][j] = unif(re);
            }

        // Initialize b
        for (int i = 0; i < ncols; ++i)
        {
            // b[i] = 1.0;
            b[i] = (double) i;
            // b[i] = unif(re);
        }

        // Allocate space for c, the answer
        c.reserve(nrows);
        c.resize(nrows);
    }
    // Worker part
    else
    {
        // Allocate space for 1 row of A
        Arow.reserve(ncols);
        Arow.resize(ncols);
    }

    // send b to every worker process, note b is a vector so b and &b[0] not same
    MPI_Bcast(&b[0], ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Output(int output, vector<double> &c, MPI_Obj &the_mpi)
{
    if (the_mpi.rank == 0 && output == 1)
    {
        cout << "( " << c[0];
        for (int i = 1; i < c.size(); ++i)
            cout << ", " << c[i];
        cout << ")\n";
    }
}

int main(int argc, char **argv)
{
    // Data File
    // ofstream datafile("/home/jared/Desktop/mv-timings.txt", ios_base::app);
    ofstream datafile("mv-timings.txt", ios_base::app);

    // initialize MPI
    MPI_Obj the_mpi(argc, argv);
    if (the_mpi.size < 2)
        MPI_Abort(MPI_COMM_WORLD, 1);

    // determine/distribute size of arrays here
    int output = 1, nrows = 0, ncols = 0;
    GetArraySize(output, nrows, ncols, the_mpi, argc, argv);
    if (the_mpi.size - 1 > nrows)
    {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    boost::multi_array<double, 2> A;
    vector<double> b(ncols);
    vector<double> c;
    vector<double> Arow;
    SetupArrays(nrows, ncols, A, b, c, Arow, the_mpi);

    MPI_Status status;

    // Timing variables
    double calc_time = 0, avg_time, total_time;

    // Boss part
    if (the_mpi.rank == 0)
    {
        total_time = MPI_Wtime();
        // send one row to each worker tagged with row number, assume size<nrows
        int rowsent = 1;
        for (int i = 1; i < the_mpi.size; i++)
        {
            MPI_Send(&A[rowsent - 1][0], ncols, MPI_DOUBLE, i, rowsent, MPI_COMM_WORLD);
            rowsent++;
        }

        for (int i = 0; i < nrows; i++)
        {
            double ans;
            MPI_Recv(&ans, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int sender = status.MPI_SOURCE;
            int row = status.MPI_TAG - 1;
            c[row] = ans;
            if (rowsent - 1 < nrows)
            { // send new row
                MPI_Send(&A[rowsent - 1][0], ncols, MPI_DOUBLE, sender, rowsent, MPI_COMM_WORLD);
                rowsent++;
            }
            else
            { // tell sender no more work to do via a 0 TAG
                MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        // Get a row of A
        MPI_Recv(&Arow[0], ncols, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        while (status.MPI_TAG != 0)
        {
            // work out Arow.b
            double ans = 0.0;

            calc_time -= MPI_Wtime();
            for (int i = 0; i < ncols; i++)
                ans += Arow[i] * b[i];
            calc_time += MPI_Wtime();

            // Send answer of Arow.b back to boss and get another row to work on
            MPI_Send(&ans, 1, MPI_DOUBLE, 0, status.MPI_TAG, MPI_COMM_WORLD);
            MPI_Recv(&Arow[0], ncols, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }

    if (the_mpi.rank == 0)
        total_time = MPI_Wtime() - total_time;

    MPI_Reduce(&calc_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (the_mpi.rank == 0)
    {
        avg_time /= (the_mpi.size - 1); // Boss node doesn't do any of the calculations
        printf("Average calculation time = %.10f\n", avg_time);
        printf("Total time = %.10f\n", total_time);

        datafile << fixed << setprecision(10);
        datafile << nrows << " " << ncols << " " << the_mpi.size << " " << avg_time << " " << total_time << endl;
    }

    // output c here on Boss node
    Output(output, c, the_mpi);
}
