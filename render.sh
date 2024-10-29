#!/bin/bash

# Build the code 
cmake -B build
cmake --build build --config release

# render the image

# CPU ray tracer
./build/cpuRayTracer

# CUDA ray tracer
