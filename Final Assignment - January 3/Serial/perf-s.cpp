#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <boost/multi_array.hpp>
#include <mpi.h>

using namespace std;

typedef boost::multi_array<bool, 2> Tgrid;

class Grid {
    public:
        int num_cells_x, num_cells_y;
        // Cell data
        // Keep two grids, one for current state and one for previous state
        // Instead of copying new to old in updateGrid, swap the pointers
        // **NOTE** The pointer p_cells1 may not always point to cells1
        // likewise for p_cells2
        Tgrid* p_cells1;
        Tgrid* p_cells2;
        Tgrid cells1;
        Tgrid cells2;
    
        Grid(int num_cells_x, int num_cells_y) {
            // Create arrays and pointers to the arrays
            this->num_cells_x = num_cells_x;
            this->num_cells_y = num_cells_y;
            this->cells1.resize(boost::extents[num_cells_y][num_cells_x]);
            this->cells2.resize(boost::extents[num_cells_y][num_cells_x]);
            this->p_cells1 = &cells1;
            this->p_cells2 = &cells2;

            // Default initialization
            for (int i = 0; i < this->num_cells_y; i++) {
                for (int j = 0; j < this->num_cells_x; j++) {
                    (*this->p_cells1)[i][j] = false;
                }
            }
        }

        Grid(int num_cells_x, int num_cells_y, string filename) {
            // Create arrays and pointers to the arrays
            this->num_cells_x = num_cells_x;
            this->num_cells_y = num_cells_y;
            this->cells1.resize(boost::extents[num_cells_x][num_cells_y]);
            this->cells2.resize(boost::extents[num_cells_x][num_cells_y]);
            this->p_cells1 = &cells1;
            this->p_cells2 = &cells2;

            ifstream file(filename);

            // Initialize from file
            char c;
            for (int i = 0; i < this->num_cells_y; i++) {
                for (int j = 0; j < this->num_cells_x; j++) {
                    file >> c;
                    (*this->p_cells1)[i][j] = (c == '1');
                }
            }
        }

        void updateGrid() {
            // Swap the grids
            swap(this->p_cells1, this->p_cells2);

            // Update the grid
            for (int i = 0; i < this->num_cells_y; i++) {
                for (int j = 0; j < this->num_cells_x; j++) {
                    (*this->p_cells1)[i][j] = this->updateCell((*this->p_cells2)[i][j], this->getNeighbours(i, j));
                }
            }
        }

        void exportToFile(string filename) {
            ofstream file;
            stringstream s;
            file.open(filename);
            for (int i = 0; i < this->num_cells_y; i++) {
                for (int j = 0; j < this->num_cells_x; j++) {
                    s << (*this->p_cells1)[i][j];
                    if (j != this->num_cells_x - 1) s << " ";
                }
                if (i != this->num_cells_y - 1) s << endl;
            }
            file << s.rdbuf();
            file.close();
        }

        void print() {
            for (int i = 0; i < this->num_cells_y; i++) {
                for (int j = 0; j < this->num_cells_x; j++) {
                    cout << ((*this->p_cells1)[i][j] ? "1" : "0");
                }
                cout << endl;
            }
            cout << endl;
        }

        void reset() {
            for (int i = 0; i < this->num_cells_y; i++) {
                for (int j = 0; j < this->num_cells_x; j++) {
                    (*this->p_cells1)[i][j] = false;
                }
            }
        }

        void randomGrid() {
            for (int i = 0; i < this->num_cells_y; i++) {
                for (int j = 0; j < this->num_cells_x; j++) {
                    (*this->p_cells1)[i][j] = (rand() % 2 == 0);
                }
            }
        }

    private:
        bool updateCell(bool alive, int neighbours) {
            // If the cell is alive
            if (alive) {
                // If the cell has less than 2 or more than 3 neighbours, it dies
                if (neighbours < 2 || neighbours > 3) {
                    return false;
                }
                // Otherwise it stays alive
                return true;
            }
            // If the cell is dead and has exactly 3 neighbours, it comes alive
            if (neighbours == 3) {
                return true;
            }
            // Otherwise it stays dead
            return false;
        }

        int getNeighbours(int row, int col) {
            int neighbours = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    if (i == 0 && j == 0) continue;
                    if (row + i < 0 || row + i >= this->num_cells_y) continue;
                    if (col + j < 0 || col + j >= this->num_cells_x) continue;
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

    Grid cells = init_from_file ? Grid(num_cells_x, num_cells_y, "init.txt") : Grid(num_cells_x, num_cells_y);
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

    if (output_to_file) {
        file.open("data.csv", ios::app);
        if (!file.is_open()) {
            cout << "Error opening file" << endl;
            exit(1);
        }
        file << fixed << setprecision(10);
        file << num_cells_x << ", " << num_cells_y << ", " << iterations << ", " << time << ", " << average_frame_time << endl;
    } else {
        printf("Average frame time: %f\n", average_frame_time);
        printf("Total time: %f\n", time);
    }

    return 0;
}