#include "visualiser.hpp"
#include <raylib.h>

// Draws the grid to the screen
void draw_grid(const std::vector<int>& grid, int width, int height, int cell_size) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int cell = grid[y * width + x];

            Color colour = BLACK;

            if (cell == 1) {
                colour = GREEN;
            } else if (cell == 2) {
                colour = RED;
            }

            DrawRectangle(
                x * cell_size,
                y * cell_size,
                cell_size,
                cell_size,
                colour
            );
        }
    }
}