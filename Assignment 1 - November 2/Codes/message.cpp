/* This program calculates the dot product of two vectors read from a file.
*  The main process reads the file and distributes the data to each process.
*  Each process can then begin calculating while the main process reads the
*  chunk of data (the main process is responsible for the last chunk).
*  The results are the collected on the main process and the final dot product
*  is calculated as the sum of the partial dot products.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>

class MPI_stuff 
{
    public:
    int size;
    int rank;

    MPI_stuff(int &argc, char** &argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    ~MPI_stuff()
    {
        MPI_Finalize();
    }
};

// Get the number of rows of data in the file from first row of file
int GetNumberElements(std::ifstream &fin) {
    int n;
    if (fin.is_open())
        fin >> n;
    else { // no file to read from
        std::cout << "Input File not found\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return n;
}

// Read and distribute n elements of a two column data file accross MPI Processes
void ReadArrays(std::ifstream &fin, std::vector<long double> &a, std::vector<long double> &b, int size, const MPI_stuff &the_mpi) {
    MPI_Status status;

    if (the_mpi.rank == 0) {
        if(fin.is_open()) {
            // Main process reads data and immediately sends to another process
            for (int i = 1; i < the_mpi.size; i++) {
                for(int j = 0; j < size; j++) {
                    fin >> a[j] >> b[j];
                }
                MPI_Send(&a[0], size, MPI_LONG_DOUBLE, i, 10, MPI_COMM_WORLD);
                MPI_Send(&b[0], size, MPI_LONG_DOUBLE, i, 20, MPI_COMM_WORLD);
            }
            // After other processes are saturated, read the remaining data
            int j = 0;
            while(!fin.eof()) {
                fin >> a[j] >> b[j];
                j++;
            }
        } else { // fp null means no file to read from
            std::cout << "Input File not found\n";
            MPI_Abort(MPI_COMM_WORLD,-1);
        }
    } else {
        MPI_Recv(&a[0], size, MPI_LONG_DOUBLE, 0, 10, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], size, MPI_LONG_DOUBLE, 0, 20, MPI_COMM_WORLD, &status);
    }
}

// Dot product of two vectors owned and fully stored on Boss node
float DotProduct(std::vector<long double> &a, std::vector<long double> &b, int size) { 

    // Work out the sum on each process
    long double sum=0, Gsum=0;
    for(int i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }

    // Collect resuls in Gsum
    MPI_Reduce(&sum, &Gsum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return Gsum;
}

int main(int argc, char** argv) {
    MPI_stuff the_mpi(argc, argv);

    std::ifstream fin;
    int n = 0;
    if(the_mpi.rank == 0) {
        fin.open("DotData.txt");
        n = GetNumberElements(fin);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int size = n / the_mpi.size;

    std::vector<long double> a, b;
    if(the_mpi.rank == 0){
        // Main process may require a slightly larger vector if n % size != 0
        a.reserve(size + n%the_mpi.size);  a.resize(size + n%the_mpi.size);
        b.reserve(size + n%the_mpi.size);  b.resize(size + n%the_mpi.size);
    } else {
        a.reserve(size);  a.resize(size);
        b.reserve(size);  b.resize(size);
    }

    ReadArrays(fin, a, b, size, the_mpi);

    float adotb = DotProduct(a, b, size);
    if(the_mpi.rank == 0) {
        std::cout << "The inner product is " << adotb << std::endl;
    }

    return 0;
}