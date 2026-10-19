#include "../include/benchmark.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
    // Grid sizes used for the benchmark runs
    const std::vector<int> BENCHMARK_SIZES = {100, 400, 800, 1000, 1200, 1600, 2000};

    // Prints the averaged timings in a simple table
    void print_benchmark_summary(
        const std::string& device_name,
        const std::vector<AveragedMetrics>& results
    ) {
        std::cout << "\nBenchmark summary for " << device_name << '\n';
        std::cout << "Fixed steps per run: " << BENCHMARK_STEPS
                  << " | Repeats per size: " << BENCHMARK_REPEATS << "\n\n";

        std::cout << std::left
                  << std::setw(12) << "Grid"
                  << std::setw(16) << "Init(ms)"
                  << std::setw(20) << "Spread Total(ms)"
                  << std::setw(16) << "Step Avg(ms)"
                  << std::setw(16) << "Total(ms)"
                  << '\n';

        std::cout << std::string(80, '-') << '\n';
        std::cout << std::fixed << std::setprecision(4);

        for (const AveragedMetrics& row : results) {
            const std::string grid_label =
                std::to_string(row.grid_size) + "x" + std::to_string(row.grid_size);

            std::cout << std::left
                      << std::setw(12) << grid_label
                      << std::setw(16) << row.avg_init_kernel_ms
                      << std::setw(20) << row.avg_spread_kernel_total_ms
                      << std::setw(16) << row.avg_spread_step_ms
                      << std::setw(16) << row.avg_total_runtime_ms
                      << '\n';
        }

        std::cout << '\n';
    }
}

void run_benchmark_mode(
    const OpenCLObjects& ocl,
    const KernelBundle& kernels
) {
    const std::string device_name = get_device_name(ocl.device);

    std::vector<AveragedMetrics> averaged_results;
    averaged_results.reserve(BENCHMARK_SIZES.size());

    for (int size : BENCHMARK_SIZES) {
        double init_sum = 0.0;
        double spread_sum = 0.0;
        double total_sum = 0.0;

        for (int repeat = 0; repeat < BENCHMARK_REPEATS; ++repeat) {
            SimulationConfig config;
            config.width = size;
            config.height = size;
            config.num_steps = BENCHMARK_STEPS;

            // Use different seeds each repeat so the runs are not identical
            config.init_seed = INIT_SEED + static_cast<cl_uint>(repeat * 1009u);
            config.step_seed = STEP_SEED + static_cast<cl_uint>(repeat * 9176u);

            std::vector<int> host_grid;
            const RunMetrics metrics = run_non_visual_simulation(ocl, kernels, config, host_grid);

            init_sum += metrics.init_kernel_ms;
            spread_sum += metrics.spread_kernel_total_ms;
            total_sum += metrics.total_runtime_ms;
        }

        AveragedMetrics averaged;
        averaged.grid_size = size;
        averaged.avg_init_kernel_ms = init_sum / static_cast<double>(BENCHMARK_REPEATS);
        averaged.avg_spread_kernel_total_ms = spread_sum / static_cast<double>(BENCHMARK_REPEATS);
        averaged.avg_spread_step_ms =
            averaged.avg_spread_kernel_total_ms / static_cast<double>(BENCHMARK_STEPS);
        averaged.avg_total_runtime_ms = total_sum / static_cast<double>(BENCHMARK_REPEATS);

        averaged_results.push_back(averaged);
    }

    print_benchmark_summary(device_name, averaged_results);
}