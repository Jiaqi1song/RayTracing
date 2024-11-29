#!/bin/bash

cmake -B build
cmake --build build
./build/RayTracingGPU > image.ppm