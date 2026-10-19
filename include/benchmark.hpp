#pragma once

#include "simulation.hpp"

// Runs benchmark mode
// Tries the simulation with multiple grid sizes and prints
// average timings so CPU and GPU results are easy to compare
void run_benchmark_mode(
    const OpenCLObjects& ocl,
    const KernelBundle& kernels
);