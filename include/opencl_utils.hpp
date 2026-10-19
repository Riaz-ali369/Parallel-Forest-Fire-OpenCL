#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <string>

// Simple container for OpenCL stuff used across the program
// Set up once in main and passed around where needed
struct OpenCLObjects {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
};

OpenCLObjects create_opencl_objects(cl_device_type device_type);
std::string load_kernel_source(const std::string& file_path);
cl_program build_program_from_file(const std::string& file_path, const OpenCLObjects& ocl);