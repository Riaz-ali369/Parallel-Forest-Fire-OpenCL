#include "simulation.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Adds the OpenCL error code to the message
[[noreturn]] void throw_opencl_error(const std::string& message, cl_int err) {
    throw std::runtime_error(message + " (OpenCL error code: " + std::to_string(err) + ")");
}

// Reads an integer and checks the minimum allowed value
int read_int_or_throw(const std::string& prompt, int min_value) {
    int value = 0;
    std::cout << prompt;
    std::cin >> value;

    if (!std::cin) {
        throw std::runtime_error("Input error: expected an integer value.");
    }

    if (value < min_value) {
        throw std::runtime_error("Input error: value is out of range.");
    }

    return value;
}

// Converts an OpenCL event duration from ns to ms
double event_duration_ms(cl_event event) {
    if (!event) {
        throw std::runtime_error("Profiling error: null OpenCL event.");
    }

    cl_ulong start = 0;
    cl_ulong end = 0;

    cl_int err = clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong),
        &start,
        nullptr
    );
    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to read event start time", err);
    }

    err = clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        &end,
        nullptr
    );
    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to read event end time", err);
    }

    return static_cast<double>(end - start) * 1e-6;
}

// Gets the device name for output and summaries
std::string get_device_name(cl_device_id device) {
    size_t size = 0;
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size);
    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to query OpenCL device name size", err);
    }

    std::vector<char> name(size);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, size, name.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to query OpenCL device name", err);
    }

    return std::string(name.data());
}

// Returns true if any cell is still burning
bool grid_has_burning_cells(const std::vector<int>& grid) {
    for (int cell : grid) {
        if (cell == 2) {
            return true;
        }
    }
    return false;
}

// Counts empty, tree, and burning cells
GridCounts count_grid_states(const std::vector<int>& grid) {
    GridCounts counts;

    for (int cell : grid) {
        if (cell == 0) {
            counts.empty++;
        } else if (cell == 1) {
            counts.tree++;
        } else if (cell == 2) {
            counts.burning++;
        }
    }

    return counts;
}

// Picks a cell size that keeps the grid visible on screen
int choose_cell_size(int width, int height) {
    const int max_window_dimension = 900;
    const int largest_dimension = (width > height) ? width : height;

    int cell_size = max_window_dimension / largest_dimension;

    if (cell_size < 2) {
        cell_size = 2;
    }
    if (cell_size > 40) {
        cell_size = 40;
    }

    return cell_size;
}

// ---------------------------
// Console input helpers
// ---------------------------

cl_device_type ask_device_type() {
    std::string choice;
    std::cout << "Select device type (cpu/gpu): ";
    std::cin >> choice;

    if (!std::cin) {
        throw std::runtime_error("Input error while reading device type.");
    }

    if (choice == "cpu") {
        return CL_DEVICE_TYPE_CPU;
    }

    if (choice == "gpu") {
        return CL_DEVICE_TYPE_GPU;
    }

    throw std::runtime_error("Invalid device type. Enter cpu or gpu.");
}

std::string ask_mode() {
    std::string mode;
    std::cout << "Select mode (visual/benchmark): ";
    std::cin >> mode;

    if (!std::cin) {
        throw std::runtime_error("Input error while reading mode.");
    }

    if (mode == "visual" || mode == "benchmark") {
        return mode;
    }

    throw std::runtime_error("Invalid mode. Enter visual or benchmark.");
}

VisualConfig ask_visual_config() {
    VisualConfig config;
    config.grid_size = read_int_or_throw("Enter visual grid size n for an n x n grid: ", 1);
    config.num_steps = VISUAL_DEFAULT_STEPS;
    return config;
}

// ------------------------------------------
// OpenCL resource creation and cleanup
// ------------------------------------------

// Creates the two buffers used for double buffering
SimulationBuffers create_simulation_buffers(const OpenCLObjects& ocl, int total_cells) {
    SimulationBuffers buffers;
    cl_int err = CL_SUCCESS;

    buffers.current_grid = clCreateBuffer(
        ocl.context,
        CL_MEM_READ_WRITE,
        sizeof(int) * total_cells,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to create current grid buffer", err);
    }

    buffers.next_grid = clCreateBuffer(
        ocl.context,
        CL_MEM_READ_WRITE,
        sizeof(int) * total_cells,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) {
        clReleaseMemObject(buffers.current_grid);
        buffers.current_grid = nullptr;
        throw_opencl_error("Failed to create next grid buffer", err);
    }

    return buffers;
}

void release_simulation_buffers(SimulationBuffers& buffers) {
    if (buffers.next_grid) {
        clReleaseMemObject(buffers.next_grid);
        buffers.next_grid = nullptr;
    }

    if (buffers.current_grid) {
        clReleaseMemObject(buffers.current_grid);
        buffers.current_grid = nullptr;
    }
}

// Keep kernel setup out of main
KernelBundle create_kernels(cl_program program) {
    KernelBundle kernels;
    cl_int err = CL_SUCCESS;

    kernels.init_kernel = clCreateKernel(program, "init_forest", &err);
    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to create init_forest kernel", err);
    }

    kernels.spread_kernel = clCreateKernel(program, "spread_step", &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernels.init_kernel);
        kernels.init_kernel = nullptr;
        throw_opencl_error("Failed to create spread_step kernel", err);
    }

    return kernels;
}

void release_kernel_bundle(KernelBundle& kernels) {
    if (kernels.spread_kernel) {
        clReleaseKernel(kernels.spread_kernel);
        kernels.spread_kernel = nullptr;
    }

    if (kernels.init_kernel) {
        clReleaseKernel(kernels.init_kernel);
        kernels.init_kernel = nullptr;
    }
}

// Releases the main OpenCL objects
void release_opencl_resources(cl_program& program, OpenCLObjects& ocl) {
    if (program) {
        clReleaseProgram(program);
        program = nullptr;
    }

    if (ocl.queue) {
        clReleaseCommandQueue(ocl.queue);
        ocl.queue = nullptr;
    }

    if (ocl.context) {
        clReleaseContext(ocl.context);
        ocl.context = nullptr;
    }

    ocl.device = nullptr;
    ocl.platform = nullptr;
}

// ----------------------------
// Device read-back helpers
// ---------------------------

// Copies the grid back from device to host memory
void read_grid_from_device(
    cl_command_queue queue,
    cl_mem grid_buffer,
    int total_cells,
    std::vector<int>& host_grid
) {
    const cl_int err = clEnqueueReadBuffer(
        queue,
        grid_buffer,
        CL_TRUE,
        0,
        sizeof(int) * total_cells,
        host_grid.data(),
        0,
        nullptr,
        nullptr
    );

    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to read grid back from device", err);
    }
}

// -------------------------
// Kernel launch helpers
// -------------------------

// Runs the init kernel and returns the kernel time
double enqueue_init_kernel(
    cl_kernel init_kernel,
    cl_command_queue queue,
    cl_mem current_grid,
    int total_cells,
    float prob_tree,
    float prob_burning,
    cl_uint init_seed
) {
    cl_int err = CL_SUCCESS;

    err  = clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &current_grid);
    err |= clSetKernelArg(init_kernel, 1, sizeof(int), &total_cells);
    err |= clSetKernelArg(init_kernel, 2, sizeof(float), &prob_tree);
    err |= clSetKernelArg(init_kernel, 3, sizeof(float), &prob_burning);
    err |= clSetKernelArg(init_kernel, 4, sizeof(cl_uint), &init_seed);

    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to set init_forest kernel arguments", err);
    }

    const size_t global_size = static_cast<size_t>(total_cells);
    cl_event init_event = nullptr;

    err = clEnqueueNDRangeKernel(
        queue,
        init_kernel,
        1,
        nullptr,
        &global_size,
        nullptr,
        0,
        nullptr,
        &init_event
    );
    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to enqueue init_forest kernel", err);
    }

    // Wait before reading profiling info
    clFinish(queue);

    double init_ms = 0.0;
    try {
        init_ms = event_duration_ms(init_event);
    }
    catch (...) {
        if (init_event) {
            clReleaseEvent(init_event);
        }
        throw;
    }

    clReleaseEvent(init_event);
    return init_ms;
}

// Runs one spread step and returns the kernel time
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
) {
    cl_int err = CL_SUCCESS;
    const int total_cells = width * height;
    const size_t global_size = static_cast<size_t>(total_cells);

    err  = clSetKernelArg(spread_kernel, 0, sizeof(cl_mem), &current_grid);
    err |= clSetKernelArg(spread_kernel, 1, sizeof(cl_mem), &next_grid);
    err |= clSetKernelArg(spread_kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(spread_kernel, 3, sizeof(int), &height);
    err |= clSetKernelArg(spread_kernel, 4, sizeof(float), &prob_immune);
    err |= clSetKernelArg(spread_kernel, 5, sizeof(float), &prob_lightning);
    err |= clSetKernelArg(spread_kernel, 6, sizeof(cl_uint), &step_seed);
    err |= clSetKernelArg(spread_kernel, 7, sizeof(int), &step);

    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to set spread_step kernel arguments", err);
    }

    cl_event step_event = nullptr;
    err = clEnqueueNDRangeKernel(
        queue,
        spread_kernel,
        1,
        nullptr,
        &global_size,
        nullptr,
        0,
        nullptr,
        &step_event
    );
    if (err != CL_SUCCESS) {
        throw_opencl_error("Failed to enqueue spread_step kernel", err);
    }

    clFinish(queue);

    double step_ms = 0.0;
    try {
        step_ms = event_duration_ms(step_event);
    }
    catch (...) {
        if (step_event) {
            clReleaseEvent(step_event);
        }
        throw;
    }

    clReleaseEvent(step_event);
    return step_ms;
}

// Small wrapper around the init kernel using the config values
double initialise_grid(
    const OpenCLObjects& ocl,
    cl_kernel init_kernel,
    SimulationBuffers& buffers,
    const SimulationConfig& config
) {
    const int total_cells = config.width * config.height;

    return enqueue_init_kernel(
        init_kernel,
        ocl.queue,
        buffers.current_grid,
        total_cells,
        config.prob_tree,
        config.prob_burning,
        config.init_seed
    );
}

// Retry the initial grid a few times so visual mode starts with fire
bool initialise_grid_with_burning_cells(
    const OpenCLObjects& ocl,
    cl_kernel init_kernel,
    SimulationBuffers& buffers,
    SimulationConfig& config,
    std::vector<int>& host_grid,
    double& init_kernel_ms
) {
    const int total_cells = config.width * config.height;

    for (int attempt = 0; attempt < VISUAL_MAX_INIT_ATTEMPTS; ++attempt) {
        config.init_seed = INIT_SEED + static_cast<cl_uint>(attempt * 97u);

        init_kernel_ms = initialise_grid(ocl, init_kernel, buffers, config);
        read_grid_from_device(ocl.queue, buffers.current_grid, total_cells, host_grid);

        if (grid_has_burning_cells(host_grid)) {
            return true;
        }
    }

    return false;
}

// ------------------------------------------
// Shared non visual simulation execution
// -----------------------------------------

// Runs the simulation without rendering and collects the benchmark timings
RunMetrics run_non_visual_simulation(
    const OpenCLObjects& ocl,
    const KernelBundle& kernels,
    const SimulationConfig& config,
    std::vector<int>& host_grid
) {
    const int total_cells = config.width * config.height;
    RunMetrics metrics;
    host_grid.assign(total_cells, 0);

    SimulationBuffers buffers = create_simulation_buffers(ocl, total_cells);

    try {
        const auto total_start = std::chrono::high_resolution_clock::now();

        metrics.init_kernel_ms = initialise_grid(ocl, kernels.init_kernel, buffers, config);

        for (int step = 0; step < config.num_steps; ++step) {
            metrics.spread_kernel_total_ms += enqueue_spread_kernel(
                kernels.spread_kernel,
                ocl.queue,
                buffers.current_grid,
                buffers.next_grid,
                config.width,
                config.height,
                config.prob_immune,
                config.prob_lightning,
                config.step_seed,
                step
            );

            // Swap buffers so the next step uses the new state
            cl_mem temp = buffers.current_grid;
            buffers.current_grid = buffers.next_grid;
            buffers.next_grid = temp;

            metrics.executed_steps++;
        }

        read_grid_from_device(ocl.queue, buffers.current_grid, total_cells, host_grid);

        const auto total_end = std::chrono::high_resolution_clock::now();
        metrics.total_runtime_ms =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();

        release_simulation_buffers(buffers);
        return metrics;
    }
    catch (...) {
        release_simulation_buffers(buffers);
        throw;
    }
}