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