# Question 1 - MISD

The program will generate an array of data on the main process. Next, the main process will send the data to every other process in the pool. Each process is then responsible for calculating a unique moment determined by the rank in the pool.

Code:

```C++
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
```

![[1-misd-1.png]]
1) In the first example, the data we are calculating the various moments of is an array full of 1s. As abase case, it is clear the code is working as the sum of N ones raised to any power and subsequently divided by N should indeed return one.

![[1-misd-2.png]]
2) In this second example, I have run the code with floating point numbers instead for a variety of different thread counts. I can be confident the code works by considering the value obtained for the first moment. The array contains the numbers $1$ through $N$ all divided by $N$. The sum of the first $N$ integers is $\frac{N(N+1)}{2}$, and dividing by $N^2$ in total yields approximately $\frac{1}{2}$, which is what we obtained from the code.

<div style="page-break-before:always"></div>

# Question 2 - SIMD

The program generates an array of data on the main process which is then distributed in equal chunks to the other threads. If the number of data points in the generated array is not divisible by the number of processes, the program will pad the array with zeros before sending the equally sized chunks. Each thread then calculates the 4th moment of it's share of the data. Finally, the main node collects each intermediate value from the threads and prints the 4th moment of the entire dataset.

Code:
```C++
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
```

![[2-simd-1.png]]
1) As a first test, I have created an array of all ones which we will calculate the $4$th moment of. the code properly returns $1$ as the the calculated value for the $4$th moment of all ones, in accordance with the `MISD` problem.

![[2-simd-2.png]]
2) As another test, I have filled the vector with the numbers $1$ through $N$ all divided by $N$, giving numbers between $0$ and $1$. The program outputs the an answer of $0.2$ for the forth moment, which again agrees with the result from the `MISD` problem.

Increasing the number of threads from $1$ to $8$ in this problem with $N=100,000,000$ results in a speedup of ~$1.24$x, 

<div style="page-break-before:always"></div>

# Question 3 - Message Passing (Dot Product)

The program will read a chunk of data on the main thread and send it to one of the worker threads. Once each worked thread has been saturated with an equal amount of data, the main thread reads the remaining (potentially larger) chunk of data. Each thread is then responsible for calculating the dot product of the two arrays it received. The main process then collects each partial dot product, and prints out the collective result.

Code: 
```C++
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
```

![[3-message-1.png]]
1) This first example demonstrates the code should be working. I have first generated a file containing two vectors, each filled with $10^7$ ones. The dot product of the two vectors is then exactly what we would expect. It should also be noted that the runtime with one thread and eight threads are nearly identical; this suggests that the program is I/O bottlenecked. 

![[3-message-2.png]]
2) In the second example, I have instead generated two vectors, each filled with $10^7$ random numbers between $0$ and $1$. The result looks reasonable and given the first test, we can trust the program is working. 

In this exercise, we parallelized the dot product and tried to minimize the amount of memory being used. The main process never contains the full array of data, it only reads in sequential chunks before sending it off to the worker process. In doing so, we have introduced an I/O bottleneck which currently defeats the purpose of parallelization altogether. For this program to be effective, the reading operation of the text file will need to become more efficient.

<div style="page-break-before:always"></div>

# Question 4 - Synchronization

This program demonstrates some of the commands that can be used to control synchronization in MPI. The base code uses blocking send and receive instructions while dividing a list of tasks between threads. We can then investigate the effects of different synchronization commands such as `MPI_Barrier` and `MPI_Isend` / `MPI_Irecv` by passing an argument in the command line.

Code:
```C++
/* This program distributes a number of tasks between various threads.
*  The main process listens for all other processes to announce what task each
*  process is responsible for. There is one process that takes longer to send in
*  this confirmation that the rest. We explore various methods of speeding up the
*  the program as a whole by reducing wait times.
*/

#include <iostream>
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

// Calculates Pi
void waste_your_time(int N = 1'000'000'000U) {
    double h = 1.0/N, sum = 0.0, x, pi;
    for (long i = 0; i < N; i++) {
        x = i*h;
        sum += 4.0 / (1.0 + x*x);
    }
    pi = h * sum;
    printf("%.12f\n", pi);
}

int main(int argc, char** argv) {
    MPI_stuff the_mpi(argc, argv);
    MPI_Status status;
    MPI_Request req_recv10, req_send10;
    
    // mode = 0: Base code
    // mode = 1: Barrier between loops
    // mode = 2: Unique send/recv tags
    // mode = 3: Non-blocking send/recvs with Wait statements
    // mode = 4: Non-blocking send/recvs, recvs seperated from main loop, with Barrier between loops
    // mode = 5: Same as 4, but no Barrier between loops
    int loops = 2, mode = 0;
    int task, taskx, node, tag = 10;

    if (argc == 3) {
        loops = atoi(argv[1]);
        mode = atoi(argv[2]);
    }

    for (int i = 0; i < loops; i++) {
        for (task = 1 + 10*i; task <= 10*(1 + i); task++) {

            node = task - the_mpi.size * (task / the_mpi.size);
            // 2nd Modification: Force the blocking behaviour of sends by making the tags unique
            if (mode == 2) {
                tag = 10 + node;
            }
            
            // Sending Data
            if (node == the_mpi.rank && node != 0) {   //Check to see if this is my task to do
                if (node == 2) waste_your_time();
                // 3rd Modification: Use non blocking sends and recieving to avoid waiting
                // 4th Modification: Also use non-blocking routines
                if (mode == 3 || mode == 4 || mode == 5) {
                    MPI_Isend(&task, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &req_send10); //Tell Boss I am the one
                } else {
                    MPI_Send(&task, 1, MPI_INT, 0, tag, MPI_COMM_WORLD); //Tell Boss I am the one
                }
            }

            // Recieving Data
            // Note: Modes 4 and 5 move this section of the loop to an independent loop
            if (the_mpi.rank == 0 && mode != 4 && mode != 5) {
                if (node == 0) {
                    printf("Processor %d will compute %d\n", 0, task);
                } else {
                    // 3rd Modification: Use non blocking sends and recieving to avoid waiting
                    if (mode == 3) {
                        req_recv10 = MPI_REQUEST_NULL;
                        MPI_Irecv(&taskx, 1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &req_recv10);
                        MPI_Wait(&req_recv10, &status);
                    } else {
                        MPI_Recv(&taskx, 1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
                    }
                    printf("Processor %d will compute %d\n", status.MPI_SOURCE, taskx);
                }
            }
        }

        // 4th Modification: Seperate the receiving so the main process does not get stuck waiting
        if (the_mpi.rank == 0 && (mode == 4 || mode == 5)) {
            for (task = 1 + 10*i; task <= 10*(1 + i); task++) {
                
                node = task - the_mpi.size * (task / the_mpi.size);

                if (node == 0) {
                    printf("Processor %d will compute %d\n", 0, task);
                } else {
                    req_recv10 = MPI_REQUEST_NULL;
                    MPI_Irecv(&taskx, 1, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &req_recv10);
                    MPI_Wait(&req_recv10, &status);
                    printf("Processor %d will compute %d\n", status.MPI_SOURCE, taskx);
                }
            }
        }
        
        // 1st Modification: Require each group of 10 tasks to finish before moving on
        // 4th Modification: Also prevent advancement to avoid unwanted behaviour from the Isend/Irecv 
        // (since they are open and receiving from any source)
        if (mode == 1 || mode == 4) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    return 0;
}
```

<div style="page-break-before:always"></div>

### Base Code:
![[4-sync-0.png]]
This is the code provided in lecture, adapted to C++. I get a similar output compared to the example output found in the lecture notes. I have implemented the `waste_your_time()` as a function that calculates $\pi$ (calculating $\arctan(1)$ using a finite integral approximation). 

<div style="page-break-before:always"></div>

### Modification 1:
![[4-sync-1.png]]
The first modification introduced a `MPI_Barrier()` between loops. This requires that all tasks in each block (here each block contains 10 tasks) finish before the next block of tasks can begin being evaluated. This can be useful if the second set of tasks depends on the output of the first 10 tasks.

<div style="page-break-before:always"></div>

### Modification 2:
![[4-sync-2.png]]
The second modification uses unique tag identifiers for the communications. Since we are using the blocking send / receive functions, this forces each task to be completed in order (essentially making parallelization pointless).

<div style="page-break-before:always"></div>

### Modification 3:
![[4-sync-3.png]]
Our third modification replaces the blocking send and receive instructions with instantaneous alternatives followed by a wait statement. On the sending process, we don't actually care when the message is received, so this allows each sending process to continue onto the next task immediately. The receiving process will have to wait for the message before using the to-be received data.

<div style="page-break-before:always"></div>

### Modification 4:
![[4-sync-4.png]]
This modification is similar to the third one, except we have moved the receiving instructions into a completely independent loop. This allows the main receiving process and the worker processes to loop independently. Here, we also have a `MPI_Barrier` inserted between each subgroup of (here 10) tasks.

<div style="page-break-before:always"></div>

### Modification 5:
![[4-sync-5.png]]
Finally as a bonus change, I have removed the `MPI_Barrier` between each subgroup of tasks. This will allow the worker processes to run through as many tasks as it can without waiting for each subgroup to be fully received first. This could be more efficient if the next group of tasks is independent of the previous group, and so on. One must be careful when receiving in this case as the receive calls are set to accept from any source.