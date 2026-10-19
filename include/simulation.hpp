#pragma once

#include "opencl_utils.hpp"

#include <CL/cl.h>

#include <string>
#include <vector>

// Shared types, constants and helper declarations for the simulation

constexpr float PROB_TREE = 0.8f;
constexpr float PROB_BURNING = 0.01f;
constexpr float PROB_IMMUNE = 0.3f;
constexpr float PROB_LIGHTNING = 0.001f;

constexpr cl_uint INIT_SEED = 12345u;
constexpr cl_uint STEP_SEED = 54321u;

constexpr int BENCHMARK_STEPS = 100;
constexpr int BENCHMARK_REPEATS = 3;

constexpr int VISUAL_DEFAULT_STEPS = 100;
constexpr float VISUAL_STEP_DELAY = 0.12f;
constexpr int VISUAL_MAX_INIT_ATTEMPTS = 20;
constexpr int VISUAL_OVERLAY_HEIGHT = 160;

struct SimulationConfig {
    int width = 0;
    int height = 0;
    int num_steps = 0;

    float prob_tree = PROB_TREE;
    float prob_burning = PROB_BURNING;
    float prob_immune = PROB_IMMUNE;
    float prob_lightning = PROB_LIGHTNING;

    cl_uint init_seed = INIT_SEED;
    cl_uint step_seed = STEP_SEED;
};

struct VisualConfig {
    int grid_size = 50;
    int num_steps = VISUAL_DEFAULT_STEPS;
};

struct GridCounts {
    int empty = 0;
    int tree = 0;
    int burning = 0;
};

struct RunMetrics {
    double init_kernel_ms = 0.0;
    double spread_kernel_total_ms = 0.0;
    double total_runtime_ms = 0.0;
    int executed_steps = 0;
};

struct AveragedMetrics {
    int grid_size = 0;
    double avg_init_kernel_ms = 0.0;
    double avg_spread_kernel_total_ms = 0.0;
    double avg_spread_step_ms = 0.0;
    double avg_total_runtime_ms = 0.0;
};

struct SimulationBuffers {
    cl_mem current_grid = nullptr;
    cl_mem next_grid = nullptr;
};

struct KernelBundle {
    cl_kernel init_kernel = nullptr;
    cl_kernel spread_kernel = nullptr;
};

// Input
cl_device_type ask_device_type();
std::string ask_mode();
VisualConfig ask_visual_config();
int read_int_or_throw(const std::string& prompt, int min_value);

// General helpers
[[noreturn]] void throw_opencl_error(const std::string& message, cl_int err);
double event_duration_ms(cl_event event);
std::string get_device_name(cl_device_id device);

bool grid_has_burning_cells(const std::vector<int>& grid);
GridCounts count_grid_states(const std::vector<int>& grid);
int choose_cell_size(int width, int height);

// OpenCL setup/cleanup
SimulationBuffers create_simulation_buffers(const OpenCLObjects& ocl, int total_cells);
void release_simulation_buffers(SimulationBuffers& buffers);

KernelBundle create_kernels(cl_program program);
void release_kernel_bundle(KernelBundle& kernels);

void release_opencl_resources(cl_program& program, OpenCLObjects& ocl);

// Device readback
void read_grid_from_device(
    cl_command_queue queue,
    cl_mem grid_buffer,
    int total_cells,
    std::vector<int>& host_grid
);

// Kernel helpers
double enqueue_init_kernel(
    cl_kernel init_kernel,
    cl_command_queue queue,
    cl_mem current_grid,
    int total_cells,
    float prob_tree,
    float prob_burning,
    cl_uint init_seed
);

double enqueue_spread_kernel(
    cl_kernel spread_kernel,
    cl_command_queue queue,
    cl_mem current_grid,
    cl_mem next_grid,
    int width,
    int height,
    float prob_immune,
    float prob_lightning,
    cl_uint step_seed,
    int step
);

double initialise_grid(
    const OpenCLObjects& ocl,
    cl_kernel init_kernel,
    SimulationBuffers& buffers,
    const SimulationConfig& config
);

bool initialise_grid_with_burning_cells(
    const OpenCLObjects& ocl,
    cl_kernel init_kernel,
    SimulationBuffers& buffers,
    SimulationConfig& config,
    std::vector<int>& host_grid,
    double& init_kernel_ms
);

// Shared simulation logic
RunMetrics run_non_visual_simulation(
    const OpenCLObjects& ocl,
    const KernelBundle& kernels,
    const SimulationConfig& config,
    std::vector<int>& host_grid
);