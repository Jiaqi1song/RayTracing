# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Project name and enabling CUDA
project(RayTracingGPU LANGUAGES CUDA)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify the target GPU architecture
# RTX 2080
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add the executable target using the main.cu file
add_executable(RayTracingGPU main.cu)
