#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <boost/multi_array.hpp>
#include <mpi.h>
#include <SDL2/SDL.h>

// I am aware that my class definitions do not follow typical OOP conventions
// There should really be getters and setters for things
// But I am too lazy to go and change all of that now

using namespace std;

// Define colours
#define _COLOUR(RED, GREEN, BLUE)    { RED, GREEN, BLUE, 0xFF }
#define COLOUR(RED, GREEN, BLUE)     ((SDL_ColoUr) { RED, GREEN, BLUE, 0xFF })

const SDL_Colour NO_COLOUR            = {0, 0, 0, 0};
const SDL_Colour COLOUR_BLACK         = _COLOUR(0, 0, 0);
const SDL_Colour COLOUR_WHITE         = _COLOUR(0xFF, 0xFF, 0xFF);
const SDL_Colour COLOUR_GRAY          = _COLOUR(0x64, 0x64, 0x64);
const SDL_Colour COLOUR_DARK_GRAY     = _COLOUR(0x1E, 0x1E, 0x1E);
const SDL_Colour COLOUR_LIGHT_GRAY    = _COLOUR(0xC8, 0xC8, 0xC8);
const SDL_Colour COLOUR_RED           = _COLOUR(0xF5, 0x3B, 0x56);
const SDL_Colour COLOUR_GREEN         = _COLOUR(0x01, 0x9F, 0x13);
const SDL_Colour COLOUR_BLUE          = _COLOUR(0x38, 0x95, 0xD3);
const SDL_Colour COLOUR_YELLOW        = _COLOUR(0xF7, 0xDC, 0x11);
const SDL_Colour COLOUR_ORANGE        = _COLOUR(0xFF, 0x85, 0);
const SDL_Colour COLOUR_PINK          = _COLOUR(0xFF, 0, 0xCE);
const SDL_Colour COLOUR_VIOLET        = _COLOUR(0x91, 0, 0xFF);

// Define MAX and MIN macros
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))

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
            this->cells1.resize(boost::extents[num_cells_y][num_cells_x]);
            this->cells2.resize(boost::extents[num_cells_y][num_cells_x]);
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

class SDLCamera {
    private:
        // SDL Window and renderer
        int screen_width, screen_height;
        int offset_x = 0, offset_y = 0;
        double zoom = 1;
        SDL_Window* window;
        SDL_Renderer* renderer;

        // Cell rendering options
        int default_cell_width = 15, default_cell_height = 15;
        int border_width = 1;

        // Colour options
        SDL_Colour border_colour = COLOUR_BLACK;
        SDL_Colour cell_alive_colour = COLOUR_GREEN;
        SDL_Colour cell_dead_colour = COLOUR_WHITE;
        SDL_Colour background_colour = COLOUR_WHITE;

        // Cell grid
        Grid* cells;

    public:
        SDLCamera(
            Grid* cells,
            int screen_width = 800, int screen_height = 800,
            int cell_width = 15, int cell_height = 15,
            int border_width = 1
        ) {
            this->cells = cells;

            // Initialize rendering options
            this->default_cell_width = cell_width;
            this->default_cell_height = cell_height;
            this->screen_width = screen_width;
            this->screen_height = screen_height;
            this->border_width = border_width;

            // Initialize SDL
            if(SDL_Init(SDL_INIT_VIDEO) < 0) throw "SDL could not be initialized!";

            this->window = SDL_CreateWindow(
                "Game of Life",
                SDL_WINDOWPOS_UNDEFINED,
                SDL_WINDOWPOS_UNDEFINED, 
                this->screen_width,
                this->screen_height,
                SDL_WINDOW_SHOWN
            );
            if(this->window == NULL) throw "Window could not be created!";

            this->renderer = SDL_CreateRenderer(this->window, -1, SDL_RENDERER_ACCELERATED);
            if(this->renderer == NULL) throw "Renderer could not be created!";

        }

        ~SDLCamera() {
            SDL_DestroyRenderer(this->renderer);
            SDL_DestroyWindow(this->window);
            SDL_Quit();
        }

        void move(int dx, int dy) {
            // Make sure the grid doesn't move off the screen
            if (this->offset_x + dx < this->screen_center_x()) return;
            if (this->offset_y + dy < this->screen_center_y()) return;
            if (this->offset_x + dx > -this->screen_center_x()) return;
            if (this->offset_y + dy > -this->screen_center_y()) return;

            this->offset_x += dx;
            this->offset_y += dy;
            this->renderGrid();
        }

        void zoomIn() {
            this->zoom *= 2;
            this->renderGrid();
        }

        void zoomOut() {
            this->zoom /= 2;
            this->renderGrid();
        }
    
        void renderGrid() {
            // If we are zoomed in, make sure the grid doesn't move off the screen
            if (this->offset_x < this->screen_center_x()) this->offset_x = this->screen_center_x();
            if (this->offset_y < this->screen_center_y()) this->offset_y = this->screen_center_y();
            if (this->offset_x > -this->screen_center_x()) this->offset_x = -this->screen_center_x();
            if (this->offset_y > -this->screen_center_y()) this->offset_y = -this->screen_center_y();

            // If the grid size is smaller than the screen, center it
            if (this->grid_width() < this->screen_width) this->offset_x = 0;
            if (this->grid_height() < this->screen_height) this->offset_y = 0;

            // Renders the entire grid of cells with given colours
            SDL_SetRenderDrawColor(
                this->renderer,
                this->background_colour.r,
                this->background_colour.g,
                this->background_colour.b,
                this->background_colour.a
            );
            SDL_RenderClear(this->renderer);

            // Draw the grid borders
            SDL_SetRenderDrawColor(
                this->renderer,
                this->border_colour.r,
                this->border_colour.g,
                this->border_colour.b,
                this->border_colour.a
            );
            SDL_Rect border_rect = {
                this->screen_center_x() + this->offset_x,
                this->screen_center_y() + this->offset_y,
                grid_width(),
                grid_height()
            };
            SDL_RenderFillRect(this->renderer, &border_rect);

            // Draw each cell
            for (int i = 0; i < this->cells->num_cells_y; i++) {
                for (int j = 0; j < this->cells->num_cells_x; j++) {
                    this->renderCell(
                        i, j,
                        (*this->cells->p_cells1)[i][j] ? this->cell_alive_colour : this->cell_dead_colour
                    );
                }
            }

            SDL_RenderPresent(this->renderer);
        }

        void toggleCell(int mouse_x, int mouse_y) {
            // Toggle the cell at the given mouse coordinates
            int cell_i = (mouse_y - this->screen_center_y() - this->offset_y) / this->cell_height();
            int cell_j = (mouse_x - this->screen_center_x() - this->offset_x) / this->cell_width();

            if (cell_i < 0 || cell_i >= this->cells->num_cells_y) return;
            if (cell_j < 0 || cell_j >= this->cells->num_cells_x) return;

            (*this->cells->p_cells1)[cell_i][cell_j] = !(*this->cells->p_cells1)[cell_i][cell_j];
            this->renderCell(
                cell_i, cell_j,
                (*this->cells->p_cells1)[cell_i][cell_j] ? this->cell_alive_colour : this->cell_dead_colour
            );
            SDL_RenderPresent(this->renderer);
        }

        void fitToScreen() {
            // Zooms out until the grid fits the screen
            double target_x = (double) this->screen_width / (this->default_cell_width * this->cells->num_cells_x);
            double target_y = (double) this->screen_width / (this->default_cell_height * this->cells->num_cells_y);
            this->zoom = min(target_x, target_y);
            this->renderGrid();
        }
        
        void clearCells() {
            this->cells->reset();
            this->renderGrid();
        }

        void randomGrid() {
            this->cells->randomGrid();
            this->renderGrid();
        }

        void nextFrame() {
            this->cells->updateGrid();
            this->renderGrid();
        }

    private:
        int cell_width() {
            return this->default_cell_width * this->zoom;
        }

        int cell_height() {
            return this->default_cell_height * this->zoom;
        }

        int grid_width() {
            return this->cells->num_cells_x * cell_width();
        }

        int grid_height() {
            return this->cells->num_cells_y * cell_height();
        }

        int screen_center_x() {
            return this->screen_width / 2 - this->grid_width() / 2;
        }

        int screen_center_y() {
            return this->screen_height / 2 - this->grid_height() / 2;
        }

        void renderCell(int cell_i, int cell_j, SDL_Colour cell_colour) {
            // Render a single cell at x, y
            SDL_Rect rect = {
                this->screen_center_x() + cell_j * cell_width() + offset_x + this->border_width,
                this->screen_center_y() + cell_i * cell_height() + offset_y + this->border_width,
                cell_width() - 2 * this->border_width,
                cell_height() - 2 * this->border_width
            };
            SDL_SetRenderDrawColor(this->renderer, cell_colour.r, cell_colour.g, cell_colour.b, cell_colour.a);
            SDL_RenderFillRect(this->renderer, &rect);
        }
};

void printControls() {
    printf("Welcome to Jared's Game of Life!\n");
    printf("Controls are as follows:\n");
    printf("  - Left click to toggle a cell\n");
    printf("  - Arrow keys to pan the grid\n");
    printf("  - Press 't' to randomize the grid\n");
    printf("  - Press 'r' to reset the grid\n");
    printf("  - Press 's' to save the grid to a file\n");
    printf("  - Press '=' / '-' to zoom in/out\n");
    printf("  - Press '[' / ']' to animate / pause the game\n");
    printf("  - Press '\\' to fit the grid to the window\n");
}

void keyPressed(
    bool* play,
    SDLCamera* camera,
    Grid* cells,
    SDL_Event* event
) {
    switch (event->key.keysym.sym) {
        case SDLK_UP:
            camera->move(0, 10);
            break;
        case SDLK_DOWN:
            camera->move(0, -10);
            break;
        case SDLK_LEFT:
            camera->move(10, 0);
            break;
        case SDLK_RIGHT:
            camera->move(-10, 0);
            break;
        case SDLK_EQUALS:
            camera->zoomIn();
            break;
        case SDLK_MINUS:
            camera->zoomOut();
            break;
        case SDLK_SPACE:
            camera->nextFrame();
            break;
        case SDLK_r:
            *play = false;
            camera->clearCells();
            break;
        case SDLK_t:
            camera->randomGrid();
            break;
        case SDLK_BACKSLASH:
            camera->fitToScreen();
            break;
        case SDLK_s:
            cells->exportToFile("cells.txt");
            break;
        case SDLK_LEFTBRACKET:
            *play = true;
            break;
        case SDLK_RIGHTBRACKET:
            *play = false;
            break;
    }
}

// Arguments:
// 1. Number of cells in x direction
// 2. Number of cells in y direction
// 3. Screen width (in pixels)
// 4. Screen height (in pixels)
// 5. Cell width (in pixels)
// 6. Cell height (in pixels)
// 7. Border width (in pixels)
int main(int argc, char** argv) {

    // Check for user input
    int num_cells_x = 10;
    int num_cells_y = 10;
    int init_from_file = 0;
    int frame_time = 100;
    int screen_width = 800;
    int screen_height = 800;
    int cell_width = 20;
    int cell_height = 20;
    int border_width = 1;

    if (argc != 1 && argc != 10) {
        printf("\nUsage: %s [frame_time num_cells_x num_cells_y screen_width screen_height cell_width cell_height border_width file_init]\n\n", argv[0]);
        exit(1);
    }

    if (argc == 10) {
        frame_time = atoi(argv[1]);
        num_cells_x = atoi(argv[2]);
        num_cells_y = atoi(argv[3]);
        screen_width = atoi(argv[4]);
        screen_height = atoi(argv[5]);
        cell_width = atoi(argv[6]);
        cell_height = atoi(argv[7]);
        border_width = atoi(argv[8]);
        init_from_file = atoi(argv[9]);
    }

    Grid cells = init_from_file ? Grid(num_cells_x, num_cells_y, "init.txt") : Grid(num_cells_x, num_cells_y);
    printControls();
    SDLCamera camera = SDLCamera(&cells, screen_width, screen_height, cell_width, cell_height, border_width);
    camera.fitToScreen();

    bool quit = false;
    bool play = false;
    SDL_Event e;
    double time_last_frame = 0;
    double elapsed_time = 0;

    while(!quit) {
        // Keep track of time between frames
        elapsed_time = SDL_GetTicks() - time_last_frame;
        if (play && elapsed_time > frame_time) {
            cells.updateGrid();
            camera.renderGrid();
            time_last_frame = SDL_GetTicks();
            elapsed_time = 0;
        }
        if (play) {
            SDL_WaitEventTimeout(&e, frame_time - elapsed_time);
        } else {
            SDL_WaitEvent(&e);
        }

        if (e.type == SDL_MOUSEBUTTONDOWN) {
            int mouse_x = e.button.x;
            int mouse_y = e.button.y;
            camera.toggleCell(mouse_x, mouse_y);
        }

        if (e.type == SDL_KEYDOWN) {
            keyPressed(&play, &camera, &cells, &e);
        }

        if (e.type == SDL_QUIT) quit = true;
    }

    return 0;
}