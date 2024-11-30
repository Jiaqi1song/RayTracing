#ifndef VEC_H
#define VEC_H

#include <cassert>
#include <cmath>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>

constexpr float EPSILON = 1e-8f;
constexpr float PI = 3.1415926535897932385f;

__device__ inline float degrees_to_radians(float degrees) { return degrees * PI / 180.0f; }

// Function to initialize cuRAND state for each thread
__global__ void init_random_state(curandState *state, int max_x, int max_y, unsigned long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curand_init(seed + pixel_index, 0, 0, &state[pixel_index]);
}

// Generate a random float in [0, 1)
__device__ float random_float(curandState *state)
{
    // Generate a random float between [0, 1) using cuRAND uniform distribution
    return curand_uniform(state);
}

// Generate a random float in [min, max)
__device__ float random_float(curandState *state, float min, float max)
{
    // Generate a random float between [min, max) using curand_uniform
    return min + (max - min) * curand_uniform(state);
}

__device__  int random_int(int min, int max, curandState *state) {
    // Returns a random integer in [min,max].
    return int(random_float(state, float(min), float(max+1)));
}

class vec_base
{
  protected:
    float e[3];

  public:
    __device__ vec_base() : e{0.0f, 0.0f, 0.0f} {}
    __device__ vec_base(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __device__ float operator[](int i) const
    {
        assert(i >= 0 && i <= 2);
        return e[i];
    }

    __device__ float &operator[](int i)
    {
        assert(i >= 0 && i <= 2);
        return e[i];
    }
};

class vec3 : public vec_base
{
  public:
    __device__ vec3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : vec_base(x, y, z) {}

    __device__ float x() const { return e[0]; }
    __device__ float y() const { return e[1]; }
    __device__ float z() const { return e[2]; }

    __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    __device__ vec3 &operator+=(const vec3 &v)
    {
        e[0] += v[0];
        e[1] += v[1];
        e[2] += v[2];
        return *this;
    }

    __device__ vec3 &operator-=(const vec3 &v)
    {
        e[0] -= v[0];
        e[1] -= v[1];
        e[2] -= v[2];
        return *this;
    }

    __device__ vec3 &operator*=(float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __device__ vec3 &operator/=(float t)
    {
        assert(t != 0.0f);
        float inv_t = 1.0f / t;
        e[0] *= inv_t;
        e[1] *= inv_t;
        e[2] *= inv_t;
        return *this;
    }

    __device__ float length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    __device__ float length() const { return sqrtf(length_squared()); }

    __device__ static vec3 random(curandState *state)
    {
        return vec3(random_float(state), random_float(state), random_float(state));
    }

    __device__ static vec3 random(curandState *state, float min, float max)
    {
        return vec3(random_float(state, min, max), random_float(state, min, max), random_float(state, min, max));
    }

    __device__ bool near_zero() const
    {
        return (fabsf(e[0]) < EPSILON && fabsf(e[1]) < EPSILON && fabsf(e[2]) < EPSILON);
    }
};

__device__ inline bool operator==(const vec3 &u, const vec3 &v)
{
    return fabsf(u[0] - v[0]) < EPSILON && fabsf(u[1] - v[1]) < EPSILON && fabsf(u[2] - v[2]) < EPSILON;
}

__device__ inline bool operator!=(const vec3 &u, const vec3 &v) { return !(u == v); }

__device__ inline vec3 operator+(const vec3 &u, const vec3 &v) { return vec3(u[0] + v[0], u[1] + v[1], u[2] + v[2]); }

__device__ inline vec3 operator-(const vec3 &u, const vec3 &v) { return vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]); }

__device__ inline vec3 operator*(const vec3 &u, float t) { return vec3(u[0] * t, u[1] * t, u[2] * t); }

__device__ inline vec3 operator*(float t, const vec3 &v) { return vec3(t * v[0], t * v[1], t * v[2]); }

__device__ inline vec3 operator/(const vec3 &u, float t)
{
    assert(t != 0.0f);
    float inv_t = 1.0f / t;
    return vec3(u[0] * inv_t, u[1] * inv_t, u[2] * inv_t);
}

__device__ inline float dot(const vec3 &u, const vec3 &v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

__device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
}

__device__ inline vec3 unit_vector(const vec3 &v) { return v.near_zero() ? vec3() : v / v.length(); }

__device__ inline vec3 random_unit_vector(curandState *state)
{
    float z = random_float(state, -1.0f, 1.0f);
    float phi = random_float(state, 0.0f, 2.0f * PI);
    float r = sqrtf(1.0f - z * z);
    return vec3(r * cosf(phi), r * sinf(phi), z);
}

__device__ inline vec3 random_in_unit_disk(curandState *state)
{
    float r = sqrtf(random_float(state));
    float theta = random_float(state, 0.0f, 2.0f * PI);

    float x = r * cosf(theta);
    float y = r * sinf(theta);

    return vec3(x, y, 0.0f);
}

__device__ inline vec3 random_on_hemisphere(curandState *state, const vec3 &normal)
{
    vec3 on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ inline vec3 reflect(const vec3 &v, const vec3 &n) { return v - 2.0f * dot(v, n) * n; }

__device__ inline vec3 refract(const vec3 &v, const vec3 &n, float eta)
{
    float cos_theta = fminf(dot(-v, n), 1.0f);
    vec3 r_out_perp = eta * (v + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ inline vec3 random_cosine_direction(curandState *state) {
    auto r1 = random_float(state);
    auto r2 = random_float(state);

    auto phi = 2*PI*r1;
    auto x = cosf(phi) * sqrtf(r2);
    auto y = sinf(phi) * sqrtf(r2);
    auto z = sqrtf(1-r2);

    return vec3(x, y, z);
}

class point3 : public vec_base
{
  public:
    __device__ point3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : vec_base(x, y, z) {}

    __device__ float x() const { return e[0]; }
    __device__ float y() const { return e[1]; }
    __device__ float z() const { return e[2]; }

    __device__ point3 operator-() const { return point3(-e[0], -e[1], -e[2]); }

    __device__ point3 &operator+=(const vec3 &v)
    {
        e[0] += v[0];
        e[1] += v[1];
        e[2] += v[2];
        return *this;
    }

    __device__ point3 &operator-=(const vec3 &v)
    {
        e[0] -= v[0];
        e[1] -= v[1];
        e[2] -= v[2];
        return *this;
    }

    __device__ point3 &operator*=(float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __device__ static point3 random(curandState *state)
    {
        return point3(random_float(state), random_float(state), random_float(state));
    }

    __device__ static point3 random(curandState *state, float min, float max)
    {
        return point3(random_float(state, min, max), random_float(state, min, max), random_float(state, min, max));
    }
};

__device__ inline float distance_to_origin(const point3 &p)
{
    return sqrtf(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
}

__device__ inline float distance(const point3 &p1, const point3 &p2)
{
    return sqrtf((p1.x() - p2.x()) * (p1.x() - p2.x()) + (p1.y() - p2.y()) * (p1.y() - p2.y()) +
                 (p1.z() - p2.z()) * (p1.z() - p2.z()));
}

__device__ inline bool operator==(const point3 &u, const point3 &v)
{
    return fabsf(u[0] - v[0]) < EPSILON && fabsf(u[1] - v[1]) < EPSILON && fabsf(u[2] - v[2]) < EPSILON;
}

__device__ inline bool operator!=(const point3 &u, const point3 &v) { return !(u == v); }

__device__ inline vec3 operator-(point3 p1, point3 p2) { return vec3(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]); }

__device__ inline vec3 operator+(point3 p1, point3 p2) { return vec3(p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2]); }

__device__ inline point3 operator+(point3 p, vec3 v) { return point3(p[0] + v[0], p[1] + v[1], p[2] + v[2]); }

__device__ inline point3 operator+(vec3 v, point3 p) { return point3(p[0] + v[0], p[1] + v[1], p[2] + v[2]); }

__device__ inline point3 operator-(point3 p, vec3 v) { return point3(p[0] - v[0], p[1] - v[1], p[2] - v[2]); }

#endif
