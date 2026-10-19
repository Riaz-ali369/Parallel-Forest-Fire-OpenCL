#include "visual_mode.hpp"
#include "visualiser.hpp"

#include <raylib.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace {
    // Draws the info panel under the grid
    void draw_visual_overlay(
        int grid_pixel_height,
        int window_width,
        const std::string& device_name,
        int grid_width,
        int grid_height,
        int current_step,
        int total_steps,
        const GridCounts& counts,
        bool simulation_finished,
        int extinguished_step,
        bool paused
    ) {
        DrawRectangle(0, grid_pixel_height, window_width, VISUAL_OVERLAY_HEIGHT, Fade(BLACK, 0.85f));

        DrawText(TextFormat("Device: %s", device_name.c_str()), 10, grid_pixel_height + 10, 20, WHITE);
        DrawText(TextFormat("Grid: %d x %d", grid_width, grid_height), 10, grid_pixel_height + 38, 20, WHITE);
        DrawText(TextFormat("Step: %d / %d", current_step, total_steps), 10, grid_pixel_height + 66, 20, WHITE);
        DrawText(
            TextFormat("Empty: %d  Tree: %d  Burning: %d", counts.empty, counts.tree, counts.burning),
            10,
            grid_pixel_height + 94,
            20,
            WHITE
        );

        if (simulation_finished) {
            if (extinguished_step >= 0) {
                DrawText(
                    TextFormat("Status: Fire extinguished at step %d", extinguished_step),
                    10,
                    grid_pixel_height + 122,
                    20,
                    LIGHTGRAY
                );
            } else {
                DrawText("Status: Reached maximum steps", 10, grid_pixel_height + 122, 20, LIGHTGRAY);
            }
        } else {
            DrawText(
                paused ? "Status: Paused (SPACE to resume)"
                       : "Status: Running (SPACE to pause)",
                10,
                grid_pixel_height + 122,
                20,
                LIGHTGRAY
            );
        }
    }
}

void run_visual_mode(
    const OpenCLObjects& ocl,
    const KernelBundle& kernels
) {
    const VisualConfig visual = ask_visual_config();

    SimulationConfig config;
    config.width = visual.grid_size;
    config.height = visual.grid_size;
    config.num_steps = visual.num_steps;

    const int total_cells = config.width * config.height;
    const int cell_size = choose_cell_size(config.width, config.height);
    const int grid_pixel_width = config.width * cell_size;
    const int grid_pixel_height = config.height * cell_size;
    const int window_width = grid_pixel_width;
    const int window_height = grid_pixel_height + VISUAL_OVERLAY_HEIGHT;

    std::vector<int> host_grid(total_cells, 0);
    SimulationBuffers buffers = create_simulation_buffers(ocl, total_cells);

    try {
        double init_kernel_ms = 0.0;

        // Retry a few times so visual mode starts with at least one fire
        const bool success = initialise_grid_with_burning_cells(
            ocl,
            kernels.init_kernel,
            buffers,
            config,
            host_grid,
            init_kernel_ms
        );

        if (!success) {
            throw std::runtime_error("Visual mode could not produce any initial burning cells.");
        }

        SetTraceLogLevel(LOG_WARNING);  // to clean up raylib console fluff
        InitWindow(window_width, window_height, "Forest Fire Visualisation");
        SetTargetFPS(60);

        const std::string device_name = get_device_name(ocl.device);

        int current_step = 0;
        int extinguished_step = -1;
        float elapsed_since_step = 0.0f;
        bool paused = false;
        bool simulation_finished = false;
        double spread_kernel_total_ms = 0.0;

        const auto total_start = std::chrono::high_resolution_clock::now();

        while (!WindowShouldClose()) {
            const float frame_time = GetFrameTime();

            if (IsKeyPressed(KEY_SPACE) && !simulation_finished) {
                paused = !paused;
            }

            // Step the simulation at a fixed pace so it stays readable
            if (!paused && !simulation_finished && current_step < config.num_steps) {
                elapsed_since_step += frame_time;

                if (elapsed_since_step >= VISUAL_STEP_DELAY) {
                    spread_kernel_total_ms += enqueue_spread_kernel(
                        kernels.spread_kernel,
                        ocl.queue,
                        buffers.current_grid,
                        buffers.next_grid,
                        config.width,
                        config.height,
                        config.prob_immune,
                        config.prob_lightning,
                        config.step_seed,
                        current_step
                    );

                    // Swap buffers so the next update uses the new state
                    cl_mem temp = buffers.current_grid;
                    buffers.current_grid = buffers.next_grid;
                    buffers.next_grid = temp;

                    read_grid_from_device(ocl.queue, buffers.current_grid, total_cells, host_grid);

                    current_step++;
                    elapsed_since_step = 0.0f;

                    // Stop early if the fire has died out
                    if (!grid_has_burning_cells(host_grid)) {
                        simulation_finished = true;
                        extinguished_step = current_step;
                    } else if (current_step >= config.num_steps) {
                        simulation_finished = true;
                    }
                }
            }

            const GridCounts counts = count_grid_states(host_grid);

            BeginDrawing();
            ClearBackground(DARKGRAY);
            DrawRectangle(0, 0, grid_pixel_width, grid_pixel_height, BLACK);
            draw_grid(host_grid, config.width, config.height, cell_size);
            draw_visual_overlay(
                grid_pixel_height,
                window_width,
                device_name,
                config.width,
                config.height,
                current_step,
                config.num_steps,
                counts,
                simulation_finished,
                extinguished_step,
                paused
            );
            EndDrawing();
        }

        if (IsWindowReady()) {
            CloseWindow();
        }

        const auto total_end = std::chrono::high_resolution_clock::now();
        const double total_runtime_ms =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "\nVisual run summary\n";
        std::cout << "Device: " << device_name << '\n';
        std::cout << "Grid: " << config.width << " x " << config.height << '\n';
        std::cout << "Executed steps: " << current_step << '\n';
        std::cout << "Init kernel time [ms]: " << init_kernel_ms << '\n';
        std::cout << "Spread kernel total [ms]: " << spread_kernel_total_ms << '\n';
        std::cout << "Total visual runtime [ms]: " << total_runtime_ms << '\n';

        release_simulation_buffers(buffers);
    }
    catch (...) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        release_simulation_buffers(buffers);
        throw;
    }
}