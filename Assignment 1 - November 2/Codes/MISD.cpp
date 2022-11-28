/*  MISD Program Example
*   Generates a large vector of numbers on the main MPI process
*   Which is subsequently distributed to all other processes
*   Each process then computes the moment corresponding to it's rank
*/ 

#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <mpi.h>

constexpr unsigned long int N = 100'000'000U;

int main(int argc, char** argv) {
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    long double moment = 0.0;
    std::vector<long double> numbers(N);

    // Generate some data
    // std::fill(numbers.begin(), numbers.end(), 1);
    iota(numbers.begin(), numbers.end(), 1);
    std::transform(numbers.begin(), numbers.end(), numbers.begin(), [](long double &x) { return x / N; });

    // Send it to every process
    MPI_Bcast(numbers.data(), N, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute the moment
    for (int i = 0; i < N; i++) {
        moment += pow(numbers[i], rank+1);
    }
    moment /= N;

    // Print the result
    printf("Process %d has calculated the %d moment to be %.6Lf\n", rank, rank+1, moment);

    MPI_Finalize();
}