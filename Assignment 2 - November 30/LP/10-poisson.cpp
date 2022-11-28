// Program to compute solution to -laplacian u = -F
// compile with
// run with mpirun -n 1 <executable> output_cout output_file N

#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include "boost/multi_array.hpp"

// LAPACK library files
extern "C"
{
    // general band factorize routine
    extern int dpbtrf_(char *, int *, int *, double *, int *, int *);
    // general band solve using factorization routine
    extern int dpbtrs_(char *, int *, int *, int *, double *, int *, double *, int *, int *);
}

void AbInit(int N, boost::multi_array<double, 2> &Ab)
{
    // Coefficient Matrix: initialize
    for (int i = 0; i < N * N; i++)
        for (int j = 0; j < N + 1; j++)
        {
            Ab[j][i] = 0.0;
            if (j == 0)
                Ab[j][i] = -1.0;
            if (j == N - 1 && i % N)
                Ab[j][i] = -1.0;
            if (j == N)
                Ab[j][i] = 4.0;
        }
    // std::cout << "Ab initialized \n";
    //  Printout Ab for testing, small N only
    // for (int i=0; i < N+1; i++) {
    //   std::cout << Ab[i][0];
    //   for (int j=1; j < N*N; j++) {
    //     std::cout << " " << Ab[i][j];
    //   }
    //   std::cout << "\n";
    // }
}

void RHSInitialize(int N, std::vector<double> &F, double bcx, double bcy)
{
    // RHS: fill in boundary condition values
    for (int i = 0; i < N; i++)
    {
        F[i] += bcx;             // bottom boundary
        F[N * N - i - 1] += bcx; // top boundary
        F[i * N] += bcy;         // left boundary
        F[i * N + N - 1] += bcy; // right boundary
    }
    // std::cout << "RHS initialized\n";

    // RHS: fill in actual right-hand side, some "charges", actually h^2*charge
    F[N / 4 * N + N / 2] += 0.5;
    F[3 * N / 4 * N + N / 2] += -0.5;
}

int main(int argc, char** argv)
{ 
    double time = MPI_Wtime();
    // ofstream timingfile("/home/jared/Desktop/lp-timings.txt", ios_base::app);
    std::ofstream timingfile("lp-timings.txt", std::ios_base::app);

    // ofstream datafile("/home/jared/Desktop/lp-data.txt");
    std::ofstream datafile("lp-data.txt");

    // Coefficient Matrix: declare
    int output_cout = 1, output_file = 0;
    int N = 10;    
    if (argc == 4) {
        output_cout = atoi(argv[1]);
        output_file = atoi(argv[2]);
        N = atoi(argv[3]);
    }

    int M = N + 1;
    int ABcols = N * N;
    boost::multi_array<double, 2> Ab(boost::extents[M][ABcols], boost::fortran_storage_order());

    AbInit(N, Ab); // Initialize coefficient matrix

    // Coefficient Matrix: factorize
    char uplo = 'U';
    int KD = N;
    int info;
    dpbtrf_(&uplo, &ABcols, &KD, &Ab[0][0], &M, &info);
    if (info)
    {
        std::cout << "Ab failed to factorize, info = " << info << "\n";
        exit(1);
    }

    // RHS: declare
    const double bcx = 0.0, bcy = 0.0; // boundary conditions along x and y assume same on both sides
    std::vector<double> F(N * N, 0.0);

    RHSInitialize(N, F, bcx, bcy); // set up boundary conditions and right hand side

    // Solve system
    int Bcols = 1;
    dpbtrs_(&uplo, &ABcols, &KD, &Bcols, &Ab[0][0], &M, &F[0], &ABcols, &info);
    if (info)
    {
        std::cout << "System solve failed, info = " << info << "\n";
        exit(1);
    }

    time = MPI_Wtime() - time;
    timingfile << N << " " << time << std::endl;

    // Output solution
    if (output_cout) {
        for (int i = 0; i < N; i++)
        {
            std::cout << F[i * N];
            for (int j = 1; j < N; j++)
            {
                std::cout << " " << F[i * N + j];
            }
            std::cout << "\n";
        }
    }
    if (output_file) {
        datafile << std::fixed << std::setprecision(10);
        for (int i = 0; i < N; i++)
        {
            datafile << F[i * N];
            for (int j = 1; j < N; j++)
            {
                datafile << " " << F[i * N + j];
            }
            datafile << "\n";
        }
    }

    return 0;
}
