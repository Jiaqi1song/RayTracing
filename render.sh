#!/bin/bash

# Build the code 
cmake -B build
cmake --build build --config release

# render the image

# CPU ray tracer
./build/cpuRayTracer

# Combine the ppm files into one gif
ffmpeg -framerate 7 -i ./images/animation/image%d.ppm -y ./images/animation.gif

# CUDA ray tracer
