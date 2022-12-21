#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <float.h>
#include <boost/multi_array.hpp>
#include <mpi.h>

// I am aware that my class definitions do not follow typical OOP conventions
// There should really be getters and setters for things
// But I am too lazy to go and change all of that now

using namespace std;

typedef boost::multi_array<int, 2> Tgrid;

class MPI_Obj
{
public:
    int size;
    int rank;
    double comm_time;
    int dims[2];
    int periodic[2] = {0, 0};
    int coords[2];
    int corners[4]; // 0: top left, 1: top right, 2: bottom left, 3: bottom right
    int neighbours[4]; // 0: left, 1: right, 2: top, 3: bottom
    MPI_Comm comm_grid;
    MPI_Status status;
    MPI_Datatype Tsubarray;
    MPI_Datatype Trow;
    MPI_Datatype Tcolumn;

    MPI_Obj(int &argc, char** &argv) {
        MPI_Init(&argc, &argv);

        MPI_Comm_size(MPI_COMM_WORLD, &this->size);
        this->dims[0] = (int) sqrt((double) this->size) + 2.*FLT_EPSILON;
        while (this->size % this->dims[0] != 0) {
            this->dims[0]--;
        }
        this->dims[1] = (int) ((double) this->size / this->dims[0]) + 2.*FLT_EPSILON;

        // Get the rank, size, and coordinates of the process
        MPI_Cart_create(MPI_COMM_WORLD, 2, this->dims, this->periodic, 0, &this->comm_grid);
        MPI_Comm_rank(this->comm_grid, &this->rank);
        MPI_Comm_size(this->comm_grid, &this->size);
        MPI_Cart_coords(this->comm_grid, this->rank, 2, this->coords);

        // Get the neighbours of the process
        MPI_Cart_shift(this->comm_grid, 0, 1, &this->neighbours[0], &this->neighbours[1]); // left, right
        MPI_Cart_shift(this->comm_grid, 1, 1, &this->neighbours[2], &this->neighbours[3]); // top, bottom

        // Get the corners of the process
        int temp_coords[2];

        // top left
        temp_coords[0] = this->coords[0] - 1; temp_coords[1] = this->coords[1] - 1;
        if (temp_coords[0] < 0 || temp_coords[1] < 0) {
            corners[0] = MPI_PROC_NULL;
        } else {
            MPI_Cart_rank(this->comm_grid, temp_coords, &corners[0]);
        }

        // top right
        temp_coords[0] = this->coords[0] + 1; temp_coords[1] = this->coords[1] - 1;
        if (temp_coords[0] > this->dims[0] - 1 || temp_coords[1] < 0) {
            corners[1] = MPI_PROC_NULL;
        } else {
            MPI_Cart_rank(this->comm_grid, temp_coords, &corners[1]);
        }

        // bottom left
        temp_coords[0] = this->coords[0] - 1; temp_coords[1] = this->coords[1] + 1;
        if (temp_coords[0] < 0 || temp_coords[1] > this->dims[1] - 1) {
            corners[2] = MPI_PROC_NULL;
        } else {
            MPI_Cart_rank(this->comm_grid, temp_coords, &corners[2]);
        }

        // bottom right
        temp_coords[0] = this->coords[0] + 1; temp_coords[1] = this->coords[1] + 1;
        if (temp_coords[0] > this->dims[0] - 1 || temp_coords[1] > this->dims[1] - 1) {
            corners[3] = MPI_PROC_NULL;
        } else {
            MPI_Cart_rank(this->comm_grid, temp_coords, &corners[3]);
        }
    }

    void initDataTypes(int array_width, int local_width, int local_height) {
        // Create the subarray datatype
        MPI_Type_vector(local_height, local_width, array_width, MPI_INT, &this->Tsubarray);
        MPI_Type_commit(&this->Tsubarray);

        // Create the row datatype
        MPI_Type_vector(1, local_width, local_width, MPI_INT, &this->Trow);
        MPI_Type_commit(&this->Trow);

        // Create the column datatype
        if (this->rank == 0){
            MPI_Type_vector(local_height, 1, array_width, MPI_INT, &this->Tcolumn);
        } else {
            MPI_Type_vector(local_height, 1, local_width, MPI_INT, &this->Tcolumn);
        }
        MPI_Type_commit(&this->Tcolumn);
    }

    ~MPI_Obj() {
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
        array<int, 4> boundary0; // corners: 0: top left, 1: top right, 2: bottom left, 3: bottom right
        vector<int> boundary1; // left
        vector<int> boundary2; // right
        vector<int> boundary3; // top
        vector<int> boundary4; // bottom

        Grid(MPI_Obj* mpi, int num_cells_x, int num_cells_y) {
            // Make sure the number of cells is divisible by the number of processes
            if ((num_cells_x % mpi->dims[0] != 0) || (num_cells_y % mpi->dims[1] != 0)) {
                cout << "Number of cells must be divisible by the number of processes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            //Initialize MPI
            this->mpi = mpi;

            // Create arrays and pointers to the arrays
            // The main process will hold the main arrays
            this->global_num_cells_x = num_cells_x;
            this->global_num_cells_y = num_cells_y;
            this->local_num_cells_x = num_cells_x / this->mpi->dims[0];
            this->local_num_cells_y = num_cells_y / this->mpi->dims[1];

            // Initialize the datatypes
            this->mpi->initDataTypes(this->global_num_cells_x, this->local_num_cells_x, this->local_num_cells_y);

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
            this->boundary1.reserve(this->local_num_cells_y);
            this->boundary1.resize(this->local_num_cells_y);
            this->boundary2.reserve(this->local_num_cells_y);
            this->boundary2.resize(this->local_num_cells_y);
            this->boundary3.reserve(this->local_num_cells_x);
            this->boundary3.resize(this->local_num_cells_x);
            this->boundary4.reserve(this->local_num_cells_x);
            this->boundary4.resize(this->local_num_cells_x);

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
            if ((num_cells_x % mpi->dims[0] != 0) || (num_cells_y % mpi->dims[1] != 0)) {
                cout << "Number of cells must be divisible by the number of processes" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            //Initialize MPI
            this->mpi = mpi;

            // Create arrays and pointers to the arrays
            // The main process will hold the main arrays
            this->global_num_cells_x = num_cells_x;
            this->global_num_cells_y = num_cells_y;
            this->local_num_cells_x = num_cells_x / this->mpi->dims[0];
            this->local_num_cells_y = num_cells_y / this->mpi->dims[1];

            // Initialize the datatypes
            this->mpi->initDataTypes(this->global_num_cells_x, this->local_num_cells_x, this->local_num_cells_y);

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
            this->boundary1.reserve(this->local_num_cells_y);
            this->boundary1.resize(this->local_num_cells_y);
            this->boundary2.reserve(this->local_num_cells_y);
            this->boundary2.resize(this->local_num_cells_y);
            this->boundary3.reserve(this->local_num_cells_x);
            this->boundary3.resize(this->local_num_cells_x);
            this->boundary4.reserve(this->local_num_cells_x);
            this->boundary4.resize(this->local_num_cells_x);

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

        int getProc(int cell_i, int cell_j) {
            // Get the rank of the processor at given cell coordinates
            int proc_ji[2] = {cell_j / this->local_num_cells_x, cell_i / this->local_num_cells_y};
            int proc;
            MPI_Cart_rank(this->mpi->comm_grid, proc_ji, &proc);
            return proc;
        }

        // For 2D processor array, we need to update:
        // 1. spreadGrid() ✔️
        // 2. sendBoundary() ✔️
        // 3. collectGrid() ✔️
        // 4. getNeighbours() ✔️
        // 5. globalRow() ✔️
        // 6. globalCol() ✔️
        // 7. getProc() ✔️
        void updateGrid() {          
            // We always update p_cells1 from the data in p_cells2  
            // Swap the grids
            swap(this->p_cells1, this->p_cells2);

            // Send boundary data between processes
            this->sendBoundary();
            
            // Update the grid
            for (int i = 0; i < this->local_num_cells_y; i++) {
                for (int j = 0; j < this->local_num_cells_x; j++) {
                    (*this->p_cells1)[i][j] = this->updateCell(
                        (*this->p_cells2)[i][j],
                        this->getNeighbours(i, j)
                    );
                }
            }
        }

        void spreadGrid() {
            if (this->mpi->rank == 0) this->mpi->comm_time -= MPI_Wtime();
            // Send the data to the other processes
            if (this->mpi->rank == 0) {
                for (int p = 1; p < this->mpi->size; p++) {
                    // Get the coordinates of the process
                    int p_coords[2];
                    MPI_Cart_coords(this->mpi->comm_grid, p, 2, p_coords);
                    int row_start = p_coords[1] * this->local_num_cells_y;
                    int col_start = p_coords[0] * this->local_num_cells_x;

                    // Send the subarray to the process
                    MPI_Send(
                        &(*this->p_cells1)[row_start][col_start],
                        1,
                        this->mpi->Tsubarray,
                        p,
                        0,
                        this->mpi->comm_grid
                    );
                }
            } else {
                // Receive the subarray from the main process
                MPI_Recv(
                    &(*this->p_cells1)[0][0],
                    this->local_num_cells_x * this->local_num_cells_y,
                    MPI_INT,
                    0,
                    0,
                    this->mpi->comm_grid,
                    &this->mpi->status
                );
            }
            if (this->mpi->rank == 0) this->mpi->comm_time += MPI_Wtime();
        }

        void sendBoundary() {
            if (this->mpi->rank == 0) this->mpi->comm_time -= MPI_Wtime();
            // Send corner cells to the other processes
            // Send top left, receive bottom right
            MPI_Sendrecv(
                &(*this->p_cells2)[0][0],
                1,
                MPI_INT,
                this->mpi->corners[0],
                0,
                &this->boundary0[3],
                1,
                MPI_INT,
                this->mpi->corners[3],
                0,
                this->mpi->comm_grid,
                &this->mpi->status
            );
            // Send top right, receive bottom left
            MPI_Sendrecv(
                &(*this->p_cells2)[0][this->local_num_cells_x - 1],
                1,
                MPI_INT,
                this->mpi->corners[1],
                0,
                &this->boundary0[2],
                1,
                MPI_INT,
                this->mpi->corners[2],
                0,
                this->mpi->comm_grid,
                &this->mpi->status
            );
            // Send bottom left, receive top right
            MPI_Sendrecv(
                &(*this->p_cells2)[this->local_num_cells_y - 1][0],
                1,
                MPI_INT,
                this->mpi->corners[2],
                0,
                &this->boundary0[1],
                1,
                MPI_INT,
                this->mpi->corners[1],
                0,
                this->mpi->comm_grid,
                &this->mpi->status
            );
            // Send bottom right, receive top left
            MPI_Sendrecv(
                &(*this->p_cells2)[this->local_num_cells_y - 1][this->local_num_cells_x - 1],
                1,
                MPI_INT,
                this->mpi->corners[3],
                0,
                &this->boundary0[0],
                1,
                MPI_INT,
                this->mpi->corners[0],
                0,
                this->mpi->comm_grid,
                &this->mpi->status
            );
            
            // Send edge boundary data to the other processes
            // Send left boundary to the left process, receive from the right process
            MPI_Sendrecv(
                &(*this->p_cells2)[0][0],
                1,
                this->mpi->Tcolumn,
                this->mpi->neighbours[0],
                0,
                &this->boundary2[0],
                this->local_num_cells_y,
                MPI_INT,
                this->mpi->neighbours[1],
                0,
                this->mpi->comm_grid,
                &this->mpi->status
            );
            // Send right boundary to the right process, receive from the left process
            MPI_Sendrecv(
                &(*this->p_cells2)[0][this->local_num_cells_x - 1],
                1,
                this->mpi->Tcolumn,
                this->mpi->neighbours[1],
                0,
                &this->boundary1[0],
                this->local_num_cells_y,
                MPI_INT,
                this->mpi->neighbours[0],
                0,
                this->mpi->comm_grid,
                &this->mpi->status
            );
            // Send top boundary to the top process, receive from the bottom process
            MPI_Sendrecv(
                &(*this->p_cells2)[0][0],
                1,
                this->mpi->Trow,
                this->mpi->neighbours[2],
                0,
                &this->boundary4[0],
                this->local_num_cells_x,
                MPI_INT,
                this->mpi->neighbours[3],
                0,
                this->mpi->comm_grid,
                &this->mpi->status
            );
            // Send bottom boundary to the bottom process, receive from the top process
            MPI_Sendrecv(
                &(*this->p_cells2)[this->local_num_cells_y - 1][0],
                1,
                this->mpi->Trow,
                this->mpi->neighbours[3],
                0,
                &this->boundary3[0],
                this->local_num_cells_x,
                MPI_INT,
                this->mpi->neighbours[2],
                0,
                this->mpi->comm_grid,
                &this->mpi->status
            );
            if (this->mpi->rank == 0) this->mpi->comm_time += MPI_Wtime();
        }

        void collectGrid() {
            if (this->mpi->rank == 0) this->mpi->comm_time -= MPI_Wtime();
            // Send the data to the other processes
            if (this->mpi->rank == 0) {
                for (int p = 1; p < this->mpi->size; p++) {
                    // Get the coordinates of the process
                    int p_coords[2];
                    MPI_Cart_coords(this->mpi->comm_grid, p, 2, p_coords);
                    int row_start = p_coords[1] * this->local_num_cells_y;
                    int col_start = p_coords[0] * this->local_num_cells_x;

                    // Receive the subarray from the process
                    MPI_Recv(
                        &(*this->p_cells1)[row_start][col_start],
                        1,
                        this->mpi->Tsubarray,
                        p,
                        0,
                        this->mpi->comm_grid,
                        &this->mpi->status
                    );
                }
            } else {
                // Send the subarray to the main process
                MPI_Send(
                    &(*this->p_cells1)[0][0],
                    this->local_num_cells_x * this->local_num_cells_y,
                    MPI_INT,
                    0,
                    0,
                    this->mpi->comm_grid
                );
            }
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
            return row + this->mpi->coords[1] * this->local_num_cells_y;
        }

        int globalCol(int col) {
            return col + this->mpi->coords[0] * this->local_num_cells_x;;
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

                    // If we are on a corner, use the corner data
                    if (i == -1 && j == -1 && row == 0 && col == 0) {
                        if (this->boundary0[0] == 1) neighbours++;
                        continue;
                    }
                    if (i == -1 && j == 1 && row == 0 && col == this->local_num_cells_x - 1) {
                        if (this->boundary0[1] == 1) neighbours++;
                        continue;
                    }
                    if (i == 1 && j == -1 && row == this->local_num_cells_y - 1 && col == 0) {
                        if (this->boundary0[2] == 1) neighbours++;
                        continue;
                    }
                    if (i == 1 && j == 1 && row == this->local_num_cells_y - 1 && col == this->local_num_cells_x - 1) {
                        if (this->boundary0[3] == 1) neighbours++;
                        continue;
                    }

                    // If the row / col is on the boundary, use the boundary data
                    if (row == 0 && i == -1) {
                        if (this->boundary3[col + j] == 1) neighbours++;
                        continue;
                    }
                    if (row == this->local_num_cells_y - 1 && i == 1) {
                        if (this->boundary4[col + j] == 1) neighbours++;
                        continue;
                    }
                    if (col == 0 && j == -1) {
                        if (this->boundary1[row + i] == 1) neighbours++;
                        continue;
                    }
                    if (col == this->local_num_cells_x - 1 && j == 1) {
                        if (this->boundary2[row + i] == 1) neighbours++;
                        continue;
                    }

                    // Otherwise check internally
                    if ((*this->p_cells2)[row + i][col + j] == 1) neighbours++;
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