#!/bin/bash

# Build the code and render the image

cmake --build build --config release
./build/raytracer > ./images/image.ppm