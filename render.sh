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

# Image size
image_width=600
image_height=600

# Select animation
animation=true

# Select animation method 
#   0: zoom + rotate 
#   1: translate 
#   2: bounce sphere (CPU Only)
animation_method=1

# Select acceleration method for CPU (OpenMP)
use_openmp=true;
num_threads=8;
critical_section=false;

# Select if use BVH for CPU or CUDA 
use_bvh=false;

# Number of sample and depth
samples_per_pixel=50
max_depth=10

if [ "$render" = "cpu" ]; then
    echo "Running CPU Ray Tracer..."
    ./build/cpuRayTracer $scene $samples_per_pixel $max_depth $animation $animation_method $use_openmp $use_bvh $num_threads $critical_section $image_width $image_height
    if [ "$animation" = true ]; then
        ffmpeg -framerate 7 -i ./images/animation/image%d.ppm -y ./images/animation.gif
        rm ./images/animation/*.ppm
    fi
elif [ "$render" = "cuda" ]; then
    echo "Running CUDA Ray Tracer..."
    ./build/cudaRayTracer $scene $samples_per_pixel $max_depth $use_bvh $image_width $image_height $animation $animation_method
    if [ "$animation" = true ]; then
        ffmpeg -framerate 7 -i ./images/animation/image%d.ppm -y ./images/animation_cuda.gif
        rm ./images/animation/*.ppm
    fi
fi

