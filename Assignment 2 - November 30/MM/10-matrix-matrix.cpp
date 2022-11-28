////////////////////////////////////////////
// Matrix-Matrix multiplication code AB=C //
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

void GetArraySize(
    int &output,
    int &nrows1, 
    int &nrowscols12, 
    int &ncols2, 
    MPI_Obj &the_mpi, 
    int argc, 
    char **argv
) {
    if (the_mpi.rank == 0)
    {
        if (argc  == 5) {
            output = atoi(argv[1]);
            nrows1 = atoi(argv[2]);
            nrowscols12 = atoi(argv[3]);
            ncols2 = atoi(argv[4]);
        } else {
            cout << "Please enter the number of rows for the first matrix -> ";
            cin >> nrows1;
            cout << "Please enter the number of columns/rows for the first/second matrix-> ";
            cin >> nrowscols12;
            cout << "Please enter the number of columns for the second matrix-> ";
            cin >> ncols2;            
        }
    }

    // send everyone nrows, ncols
    int buf[3] = {nrows1, nrowscols12, ncols2};
    MPI_Bcast(buf, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (the_mpi.rank != 0)
    {
        nrows1 = buf[0];
        nrowscols12 = buf[1];
        ncols2 = buf[2];
    }
}

void SetupArrays(
    int nrows1, 
    int nrowscols12, 
    int ncols2, 
    boost::multi_array<double, 2> &A, 
    boost::multi_array<double, 2> &B, 
    boost::multi_array<double, 2> &C, 
    vector<double> &Arow,
    vector<double> &Crow,
    MPI_Obj &the_mpi
) {
    uniform_real_distribution<double> unif(-1.0, 10);
    default_random_engine re;

    B.resize(boost::extents[nrowscols12][ncols2]);
    Crow.reserve(ncols2); Crow.resize(ncols2); // Main process will need to store the values temporarily

    // Boss part
    if (the_mpi.rank == 0)
    {
        // Set size of A
        A.resize(boost::extents[nrows1][nrowscols12]);
        C.resize(boost::extents[nrows1][ncols2]);

        // Initialize A
        for (int i = 0; i < nrows1; ++i) {
            for (int j = 0; j < nrowscols12; ++j) {
                // Identity
                // if (i == j)
                //     A[i][j] = 1.0;
                // else
                //     A[i][j] = 0.0;
                // Reverse Identity
                if (i == (nrowscols12 - j - 1))
                    A[i][j] = 1.0;
                else
                    A[i][j] = 0.0;
                // Random
                // A[i][j] = unif(re);
            }
        }

        // Initialize B
        for (int i = 0; i < nrowscols12; ++i) {
            for (int j = 0; j < ncols2; ++j) {
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
                // B[i][j] = unif(re);
                // Other
                if (i == j)
                    B[i][j] = (double) i + 1;
                else
                    B[i][j] = 0;
            }
        }
    } else {
        // Worker part
        // Allocate space for 1 row of A and 1 row of the answer C
        Arow.reserve(nrowscols12); Arow.resize(nrowscols12);
    }
    MPI_Bcast(&B[0][0], nrowscols12*ncols2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Output(int output, boost::multi_array<double, 2> &array, MPI_Obj &the_mpi)
{
    if (the_mpi.rank == 0 && output) {
        cout << endl << fixed << setprecision(4);

        for (int i = 0; i < array.shape()[0]; i++) {
            for (int j = 0; j < array.shape()[1]; j++) {
                cout << array[i][j];
                if (j < array.shape()[1] - 1) cout << ", ";
            }
            cout << endl;
        }
        cout << endl;

        cout << scientific;
    }
}

int main(int argc, char **argv)
{
    // Data File
    // ofstream datafile("/home/jared/Desktop/mmblas-timings.txt", ios_base::app);
    ofstream datafile("mm-timings.txt", ios_base::app);

    // initialize MPI
    MPI_Obj the_mpi(argc, argv);
    if (the_mpi.size < 2) MPI_Abort(MPI_COMM_WORLD, 1);

    // determine/distribute size of arrays here
    int output = 1, nrows1 = 0, nrowscols12 = 0, ncols2 = 0;
    GetArraySize(output, nrows1, nrowscols12, ncols2, the_mpi, argc, argv);
    if (the_mpi.size - 1 > nrows1) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    boost::multi_array<double, 2> A;
    boost::multi_array<double, 2> B;
    boost::multi_array<double, 2> C;
    vector<double> Arow;
    vector<double> Crow;
    SetupArrays(nrows1, nrowscols12, ncols2, A, B, C, Arow, Crow, the_mpi);

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
            MPI_Send(&A[rowsent - 1][0], nrowscols12, MPI_DOUBLE, i, rowsent, MPI_COMM_WORLD);
            rowsent++;
        }

        for (int i = 0; i < nrows1; i++)
        {
            MPI_Recv(&Crow[0], ncols2, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int sender = status.MPI_SOURCE;
            int row = status.MPI_TAG - 1;
            memcpy(&C[row][0], &Crow[0], ncols2 * sizeof(double));

            if (rowsent - 1 < nrows1) { 
                // send new row
                MPI_Send(&A[rowsent - 1][0], nrowscols12, MPI_DOUBLE, sender, rowsent, MPI_COMM_WORLD);
                rowsent++;
            } else { 
                // tell sender no more work to do via a 0 TAG
                MPI_Send(MPI_BOTTOM, 0, MPI_DOUBLE, sender, 0, MPI_COMM_WORLD);
            }
        }
    }
    // Worker part: compute dot products of Arow.b until done message recieved
    else
    {
        // Get a row of A
        MPI_Recv(&Arow[0], nrowscols12, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        while (status.MPI_TAG != 0)
        {
            for (int i = 0; i < ncols2; i++) {
                // work out Crow = Arow.B
                double c = 0;
                calc_time -= MPI_Wtime();
                for (int j = 0; j < nrowscols12; j++) {
                    c += Arow[j] * B[j][i];
                }
                calc_time += MPI_Wtime();
                Crow[i] = c;

            }
            // Send answer of Arow.B back to boss and get another row to work on
            MPI_Send(&Crow[0], ncols2, MPI_DOUBLE, 0, status.MPI_TAG, MPI_COMM_WORLD);
            MPI_Recv(&Arow[0], nrowscols12, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
        // cout << "Worker " << the_mpi.rank << " got kill tag\n";
    }

    if (the_mpi.rank == 0)
        total_time = MPI_Wtime() - total_time;

    MPI_Reduce(&calc_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_time /= (the_mpi.size - 1); // Boss node doesn't do any of the calculations

    if (the_mpi.rank == 0)
    {
        printf("Average calculation time = %.10f\n", avg_time);
        printf("Total time = %.10f\n", total_time);

        datafile << fixed << setprecision(10);
        datafile << nrows1 << " " << nrowscols12 << " " << ncols2 << " " << 
            the_mpi.size << " " << avg_time << " " << total_time << endl;
    }

    // output c here on Boss node
    Output(output, C, the_mpi);
}
