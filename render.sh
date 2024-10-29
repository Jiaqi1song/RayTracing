#!/bin/bash

# Build the code and render the image

cmake --build build --config release

# CPU ray tracer
./build/cpuRayTracer > ./images/image.ppm

# CUDA ray tracer