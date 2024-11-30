#!/bin/bash

# Build the code 
cmake -B build
cmake --build build 

# Set render to "cpu" for CPU ray tracer or "cuda" for CUDA ray tracer.
render=cuda

# Select the scene
#   1: first_scene
#   2: cornell_box
#   3: final_scene
scene=1

# Select animation
animation=false

# Select animation method (only support on CPU now)
#   0: zoom + rotate 
#   1: translate 
#   2: bounce sphere
animation_method=0

# Select acceleration method for CPU
use_openmp=false;
num_threads=8;
critical_section=false;

# Select if use BVH for CPU or CUDA
use_bvh=false;

# Number of sample and depth
samples_per_pixel=1000
max_depth=50

if [ "$render" = "cpu" ]; then
    echo "Running CPU Ray Tracer..."
    ./build/cpuRayTracer $scene $samples_per_pixel $max_depth $animation $animation_method $use_openmp $use_bvh $num_threads $critical_section
    if [ "$animation" = true ]; then
        ffmpeg -framerate 7 -i ./images/animation/image%d.ppm -y ./images/animation.gif
        rm ./images/animation/*.ppm
    fi
elif [ "$render" = "cuda" ]; then
    echo "Running CUDA Ray Tracer..."
    ./build/cudaRayTracer $scene $samples_per_pixel $max_depth $use_bvh > images/test_cuda.ppm
fi

