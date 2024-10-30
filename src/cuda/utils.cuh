#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <curand_kernel.h>

// Common Headers

#include "color.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;


// Constants

__device__ const float infinity = std::numeric_limits<float>::infinity();
__device__ const float pi = 3.1415926535897932385;


// Utility Functions

__device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__device__ inline float random_float(curandState *local_rand_state) {
    // Returns a random real in [0,1) using curand.
    return curand_uniform(local_rand_state);
}

__device__ inline float random_float(float min, float max, curandState *local_rand_state) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_float(local_rand_state);
}

__device__ inline int random_int(int min, int max, curandState *local_rand_state) {
    // Returns a random integer in [min,max].
    return int(random_float(min, max+1, local_rand_state));
}



#endif
