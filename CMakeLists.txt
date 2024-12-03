#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for Ray Tracing on CPU and GPU
#---------------------------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.10.0...3.27.0 )

project ( RayTracing LANGUAGES CXX CUDA )

# Set to C++17
set ( CMAKE_CXX_STANDARD          17 )
set ( CMAKE_CXX_STANDARD_REQUIRED ON )
set ( CMAKE_CXX_EXTENSIONS        ON )

# NVIDIA RTX 2080: Turing architecture (75) 
# NVIDIA RTX 3060: Ampere architecture (86)
# Tesla V100:       Volta architecture (70)
set(CMAKE_CUDA_ARCHITECTURES 75 86 )

# Source
set ( CPURENDERER
  src/cpu/cpuRenderer.cpp
  src/cpu/stb_image.h
  src/cpu/aabb.h
  src/cpu/camera.h
  src/cpu/color.h
  src/cpu/constant_medium.h
  src/cpu/hittable.h
  src/cpu/hittable_list.h
  src/cpu/interval.h
  src/cpu/material.h
  src/cpu/onb.h
  src/cpu/pdf.h
  src/cpu/perlin.h
  src/cpu/quad.h
  src/cpu/ray.h
  src/cpu/stb_image_utils.h
  src/cpu/utils.h
  src/cpu/sphere.h
  src/cpu/texture.h
  src/cpu/vec3.h
)

set ( GPURENDERER
  src/cuda/cudaRenderer.cu
  src/cuda/aabb.h
  src/cuda/camera.h
  src/cuda/color.h
  src/cuda/constant_medium.h
  src/cuda/hittable.h
  src/cuda/interval.h
  src/cuda/material.h
  src/cuda/perlin.h
  src/cuda/quad.h
  src/cuda/ray.h
  src/cuda/sphere.h
  src/cuda/texture.h
  src/cuda/vec.h
)

include_directories(src)

# Executables
add_executable(cpuRayTracer ${CPURENDERER})
add_executable(cudaRayTracer ${GPURENDERER})

# OpenMP for CPU
find_package(OpenMP)
target_link_libraries(cpuRayTracer PUBLIC OpenMP::OpenMP_CXX)