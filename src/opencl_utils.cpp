#include "../include/opencl_utils.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

OpenCLObjects create_opencl_objects(cl_device_type device_type) {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);

    if (err != CL_SUCCESS || num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found.");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to retrieve platform IDs.");
    }

    // Go through the platforms and use the first matching device
    for (cl_uint i = 0; i < num_platforms; ++i) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[i], device_type, 0, nullptr, &num_devices);

        if (err != CL_SUCCESS || num_devices == 0) {
            continue;
        }

        std::vector<cl_device_id> devices(num_devices);
        err = clGetDeviceIDs(platforms[i], device_type, num_devices, devices.data(), nullptr);

        if (err != CL_SUCCESS) {
            continue;
        }

        const cl_device_id device = devices[0];

        // Create the context for the chosen device
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL context.");
        }

        // Turn profiling on so kernel timings can be measured.
        const cl_queue_properties queue_properties[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
            0
        };

        cl_command_queue queue = clCreateCommandQueueWithProperties(
            context,
            device,
            queue_properties,
            &err
        );

        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            throw std::runtime_error("Failed to create OpenCL command queue.");
        }

        return OpenCLObjects{platforms[i], device, context, queue};
    }

    throw std::runtime_error("Requested device type not found.");
}

std::string load_kernel_source(const std::string& file_path) {
    std::ifstream file(file_path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open kernel file: " + file_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

cl_program build_program_from_file(const std::string& file_path, const OpenCLObjects& ocl) {
    const std::string source = load_kernel_source(file_path);
    const char* source_cstr = source.c_str();
    const size_t source_size = source.size();

    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(
        ocl.context,
        1,
        &source_cstr,
        &source_size,
        &err
    );

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL program from source.");
    }

    err = clBuildProgram(program, 1, &ocl.device, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        // If the build fails, grab the log so the error is easier to debug
        size_t log_size = 0;
        clGetProgramBuildInfo(program, ocl.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(
            program,
            ocl.device,
            CL_PROGRAM_BUILD_LOG,
            log_size,
            build_log.data(),
            nullptr
        );

        const std::string log(build_log.begin(), build_log.end());
        clReleaseProgram(program);

        throw std::runtime_error("OpenCL build failed:\n" + log);
    }

    return program;
}