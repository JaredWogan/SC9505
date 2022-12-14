#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <boost/multi_array.hpp>
#include <mpi.h>

using namespace std;

typedef boost::multi_array<int, 2> Tgrid;

class MPI_Obj
{
public:
    int size;
    int rank;
    double comm_time = 0;
    MPI_Status status;

    MPI_Obj(int &argc, char** &argv)
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

class Grid {
    public:
        // MPI object
        MPI_Obj* mpi;

        int global_num_cells_x, global_num_cells_y;
        int local_num_cells_x, local_num_cells_y;
        // Cell data
        // Keep two grids, one for current state and one for previous state
        // Instead of copying new to old in updateGrid, swap the pointers
        // **NOTE** The pointer p_cells1 may not always point to cells1
        // likewise for p_cells2
        Tgrid* p_cells1;
        Tgrid* p_cells2;
        Tgrid cells1;
        Tgrid cells2;

        // Create the boundary cells
        vector<int> boundary1;
        vector<int> boundary2;

        Grid(MPI_Obj* mpi, int num_cells_x, int num_cells_y) {
            // Make sure the number of cells is divisible by the number of processes
            if (num_cells_x % mpi->size != 0) {
                cout << "Number of cells must be divisible by the number of processes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            //Initialize MPI
            this->mpi = mpi;

            // Create arrays and pointers to the arrays
            // The main process will hold the main arrays
            this->global_num_cells_x = num_cells_x;
            this->global_num_cells_y = num_cells_y;
            this->local_num_cells_x = num_cells_x;
            this->local_num_cells_y = num_cells_y / this->mpi->size;

            // Main process will hold the main arrays
            // Other processes will hold the local (smaller) arrays
            if (this->mpi->rank == 0) {
                this->cells1.resize(boost::extents[this->global_num_cells_y][this->global_num_cells_x]);
                this->cells2.resize(boost::extents[this->global_num_cells_y][this->global_num_cells_x]);
            } else {
                this->cells1.resize(boost::extents[this->local_num_cells_y][this->local_num_cells_x]);
                this->cells2.resize(boost::extents[this->local_num_cells_y][this->local_num_cells_x]);
            }
            this->p_cells1 = &cells1;
            this->p_cells2 = &cells2;

            // Resize boundary arrays
            this->boundary1.reserve(this->local_num_cells_x);
            this->boundary1.resize(this->local_num_cells_x);
            this->boundary2.reserve(this->local_num_cells_x);
            this->boundary2.resize(this->local_num_cells_x);

            // Default initialization
            if (this->mpi->rank == 0) {
                for (int i = 0; i < this->global_num_cells_y; i++) {
                    for (int j = 0; j < this->global_num_cells_x; j++) {
                        (*this->p_cells1)[i][j] = 0;
                    }
                }
            }

            this->spreadGrid();
        }

        Grid(MPI_Obj* mpi, int num_cells_x, int num_cells_y, string filename) {
            // Make sure the number of cells is divisible by the number of processes
            if (num_cells_x % mpi->size != 0) {
                cout << "Number of cells must be divisible by the number of processes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            //Initialize MPI
            this->mpi = mpi;

            // Create arrays and pointers to the arrays
            // The main process will hold the main arrays
            this->global_num_cells_x = num_cells_x;
            this->global_num_cells_y = num_cells_y;
            this->local_num_cells_x = num_cells_x;
            this->local_num_cells_y = num_cells_y / this->mpi->size;

            // Main process will hold the main arrays
            // Other processes will hold the local (smaller) arrays
            if (this->mpi->rank == 0) {
                this->cells1.resize(boost::extents[this->global_num_cells_y][this->global_num_cells_x]);
                this->cells2.resize(boost::extents[this->global_num_cells_y][this->global_num_cells_x]);
            } else {
                this->cells1.resize(boost::extents[this->local_num_cells_y][this->local_num_cells_x]);
                this->cells2.resize(boost::extents[this->local_num_cells_y][this->local_num_cells_x]);
            }
            this->p_cells1 = &cells1;
            this->p_cells2 = &cells2;

            // Resize boundary arrays
            this->boundary1.reserve(this->local_num_cells_x);
            this->boundary1.resize(this->local_num_cells_x);
            this->boundary2.reserve(this->local_num_cells_x);
            this->boundary2.resize(this->local_num_cells_x);

            ifstream file(filename);

            // Initialize from file
            if (this->mpi->rank == 0) {
                char c;
                for (int i = 0; i < this->global_num_cells_y; i++) {
                    for (int j = 0; j < this->global_num_cells_x; j++) {
                        file >> c;
                        (*this->p_cells1)[i][j] = (c == '1');
                    }
                }
            }

            this->spreadGrid();
        }

        void updateGrid() {          
            // We always update p_cells1 from the data in p_cells2  
            // Swap the grids
            swap(this->p_cells1, this->p_cells2);

            // Send boundary data between processes
            this->sendBoundary();
            
            // Update the grid
            for (int i = 0; i < this->local_num_cells_y; i++) {
                for (int j = 0; j < this->local_num_cells_x; j++) {
                    (*this->p_cells1)[i][j] = this->updateCell((*this->p_cells2)[i][j], this->getNeighbours(i, j));
                }
            }
        }

        void spreadGrid() {
            if (this->mpi->rank == 0) this->mpi->comm_time -= MPI_Wtime();
            // Send the data to the other processes
            MPI_Scatter(
                &(*this->p_cells1)[0][0],
                this->local_num_cells_x * this->local_num_cells_y,
                MPI_INT,
                &(*this->p_cells1)[0][0],
                this->local_num_cells_x * this->local_num_cells_y,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );
            if (this->mpi->rank == 0) this->mpi->comm_time += MPI_Wtime();
        }

        void sendBoundary() {
            if (this->mpi->rank == 0) this->mpi->comm_time -= MPI_Wtime();
            // Send the data to the other processes
            int above = this->mpi->rank - 1;
            int below = this->mpi->rank + 1;
            if (above < 0) above = MPI_PROC_NULL;
            if (below > this->mpi->size - 1) below = MPI_PROC_NULL;

            // Send the top row to the process above
            MPI_Sendrecv(
                &(*this->p_cells2)[0][0],
                this->local_num_cells_x,
                MPI_INT,
                above,
                0,
                &this->boundary2[0],
                this->local_num_cells_x,
                MPI_INT,
                below,
                0,
                MPI_COMM_WORLD,
                &this->mpi->status
            );
            // Send the bottom row to the process below
            MPI_Sendrecv(
                &(*this->p_cells2)[this->local_num_cells_y - 1][0],
                this->local_num_cells_x,
                MPI_INT,
                below,
                0,
                &this->boundary1[0],
                this->local_num_cells_x,
                MPI_INT,
                above,
                0,
                MPI_COMM_WORLD,
                &this->mpi->status
            );
            if (this->mpi->rank == 0) this->mpi->comm_time += MPI_Wtime();
        }

        void collectGrid() {
            if (this->mpi->rank == 0) this->mpi->comm_time -= MPI_Wtime();
            // Send the data to the other processes
            MPI_Gather(
                &(*this->p_cells1)[0][0],
                this->local_num_cells_x * this->local_num_cells_y,
                MPI_INT,
                &(*this->p_cells1)[0][0],
                this->local_num_cells_x * this->local_num_cells_y,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );
            if (this->mpi->rank == 0) this->mpi->comm_time += MPI_Wtime();
        }

        void exportToFile(string filename) {
            this->collectGrid();
            if (this->mpi->rank != 0) return;
            ofstream file;
            stringstream s;
            file.open(filename);
            for (int i = 0; i < this->global_num_cells_y; i++) {    
                for (int j = 0; j < this->global_num_cells_x; j++) {
                    s << (*this->p_cells1)[i][j];
                    if (j != this->global_num_cells_x - 1) s << " ";
                }
                if (i != this->global_num_cells_y - 1) s << endl;
            }
            file << s.rdbuf();
            file.close();
        }

        void print() {
            this->collectGrid();
            if (this->mpi->rank != 0) return;
            for (int i = 0; i < this->global_num_cells_y; i++) {
                for (int j = 0; j < this->global_num_cells_x; j++) {
                    cout << ((*this->p_cells1)[i][j] ? "1" : "0");
                    if (j != this->global_num_cells_x - 1) cout << " ";
                }
                cout << endl;
            }
            cout << endl;
        }

        void reset() {
            if (this->mpi->rank == 0) {
                for (int i = 0; i < this->global_num_cells_y; i++) {
                    for (int j = 0; j < this->global_num_cells_x; j++) {
                        (*this->p_cells1)[i][j] = 0;
                    }
                }
            }
            this->spreadGrid();
        }

        void randomGrid() {
            if (this->mpi->rank == 0) {
                for (int i = 0; i < this->global_num_cells_y; i++) {
                    for (int j = 0; j < this->global_num_cells_x; j++) {
                        (*this->p_cells1)[i][j] = (rand() % 2 == 0);
                    }
                }
            }
            this->spreadGrid();
        }

    private:
        int globalRow(int row) {
            return row + this->mpi->rank * this->local_num_cells_y;
        }

        int globalCol(int col) {
            return col;
        }

        int updateCell(int alive, int neighbours) {
            // If the cell is alive
            if (alive) {
                // If the cell has less than 2 or more than 3 neighbours, it dies
                if (neighbours < 2 || neighbours > 3) {
                    return 0;
                }
                // Otherwise it stays alive
                return 1;
            }
            // If the cell is dead and has exactly 3 neighbours, it comes alive
            if (neighbours == 3) {
                return 1;
            }
            // Otherwise it stays dead
            return 0;
        }

        int getNeighbours(int row, int col) {
            int neighbours = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    // Make sure the row and column are within the grid
                    if (i == 0 && j == 0) continue;
                    if (this->globalRow(row) + i < 0 || this->globalRow(row) + i >= this->global_num_cells_y) continue;
                    if (this->globalCol(col) + j < 0 || this->globalCol(col) + j >= this->global_num_cells_x) continue;

                    // If the row is on the boundary, use the boundary data
                    if (row == 0 && i == -1) {
                        if (this->boundary1[col + j]) neighbours++;
                        continue;
                    } else if (row == this->local_num_cells_y - 1 && i == 1) {
                        if (this->boundary2[col + j]) neighbours++;
                        continue;
                    }

                    if ((*this->p_cells2)[row + i][col + j]) neighbours++;
                }
            }
            return neighbours;
        }
};

// Arguments:
// 1. Number of cells in x direction
// 2. Number of cells in y direction
// 3. Screen width (in pixels)
// 4. Screen height (in pixels)
// 5. Cell width (in pixels)
// 6. Cell height (in pixels)
// 7. Border width (in pixels)
int main(int argc, char** argv) {
    // Output file
    ofstream file;

    // Check for user input
    int iterations = 100'000;
    int num_cells_x = 10;
    int num_cells_y = 10;
    int init_from_file = 0;
    int output_to_file = 0;

    if (argc != 1 && argc != 6) {
        printf("\nUsage: %s [iterations num_cells_x num_cells_y file_init output_to_file]\n\n", argv[0]);
        exit(1);
    }
    if (argc == 6) {
        iterations = atoi(argv[1]);
        num_cells_x = atoi(argv[2]);
        num_cells_y = atoi(argv[3]);
        init_from_file = atoi(argv[4]);
        output_to_file = atoi(argv[5]);
    }

    MPI_Obj mpi = MPI_Obj(argc, argv);

    Grid cells = init_from_file ? Grid(&mpi, num_cells_x, num_cells_y, "init.txt") : Grid(&mpi, num_cells_x, num_cells_y);
    if (!init_from_file) cells.randomGrid();

    double time = MPI_Wtime();
    double average_frame_time = 0;

    for (int i = 0; i < iterations; i++) {
        average_frame_time -= MPI_Wtime();
        cells.updateGrid();
        average_frame_time += MPI_Wtime();
    }

    time = MPI_Wtime() - time;
    average_frame_time /= iterations;

    if (mpi.rank != 0) return 0;
    if (output_to_file) {
        file.open("data.csv", ios::app);
        if (!file.is_open()) {
            cout << "Error opening file" << endl;
            exit(1);
        }
        file << fixed << setprecision(10);
        file << num_cells_x << ", " << num_cells_y << ", " << mpi.size << ", " 
             << iterations << ", " << time << ", " << average_frame_time << ", "
             << mpi.comm_time << endl;
    } else {
        printf("Average frame time: %f\n", average_frame_time);
        printf("Total time: %f\n", time);
    }

    return 0;
}