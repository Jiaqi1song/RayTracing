# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(RayTracingCPU LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable target
add_executable(RayTracingCPU main.cc)

# Add OpenMP support
find_package(OpenMP)
target_link_libraries(RayTracingCPU PUBLIC OpenMP::OpenMP_CXX)