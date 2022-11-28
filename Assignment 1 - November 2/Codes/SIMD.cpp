/* This program calculates the 4th moment of an array of numbers.
*  The data is generated on the main process. Then, the data is 
*  split amongst the processes in even chunks. Each process then
*  calculates it's portion of the moment. The results are finally
* collected on the main process and the final moment is calculated.
*/ 

#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <mpi.h>

constexpr unsigned int N = 100'000'000U;
constexpr int n_moment = 4;

int main(int argc, char** argv) {
    int size, rank;
    double moment = 0.0;
    double global_moment = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int padded_N = N % size == 0 ? N : N + size - N % size;
    std::vector<double> numbers;

    // Generate some data
    if (rank == 0) {
        numbers.reserve(padded_N); // Pad the vector to be a multiple of the number of processes
        numbers.resize(padded_N);
        // std::transform(numbers.begin(), numbers.end(), numbers.begin(), [](double &x) { return 1; });
        iota(numbers.begin(), numbers.end(), 1);
        std::transform(numbers.begin(), numbers.end(), numbers.begin(), [](double &x) { return x / N; });
    } else {
        numbers.reserve(padded_N / size); // The other processes only need to reserve space for their part of the vector
        numbers.resize(padded_N / size);
    }
    
    // Send data to every process
    MPI_Scatter(numbers.data(), padded_N / size, MPI_DOUBLE, numbers.data(), padded_N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute the moment
    for (int i = 0; i < padded_N / size; i++) {
        moment += pow(numbers[i], n_moment);
    }
    moment /= N;

    // Collect each process' moment
    MPI_Reduce(&moment, &global_moment, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print the result
    if (rank == 0) {
        printf("The 4th moment is %f\n", global_moment);
    }

    MPI_Finalize();
}