#include "../include/benchmark.hpp"
#include "../include/opencl_utils.hpp"
#include "../include/simulation.hpp"
#include "../include/visual_mode.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    cl_program program = nullptr;
    OpenCLObjects ocl{};
    KernelBundle kernels{};

    try {
        // Get device and mode from user
        const cl_device_type device_type = ask_device_type();
        const std::string mode = ask_mode();

        // Set up OpenCL and build kernels
        ocl = create_opencl_objects(device_type);
        program = build_program_from_file("kernels/forest_fire.cl", ocl);
        kernels = create_kernels(program);

        if (mode == "visual") {
            run_visual_mode(ocl, kernels);
        } else {
            run_benchmark_mode(ocl, kernels);
        }

        release_kernel_bundle(kernels);
        release_opencl_resources(program, ocl);
    }
    catch (const std::exception& error) {
        std::cerr << "\nError: " << error.what() << '\n';

        release_kernel_bundle(kernels);
        release_opencl_resources(program, ocl);
        return 1;
    }

    return 0;
}