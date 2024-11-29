#ifndef RAY_H
#define RAY_H

#include "vec.h"

class ray
{
  private:
    point3 orig;
    vec3 dir;

  public:
    __device__ ray() : orig(point3()), dir(vec3()) {}
    __device__ ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(unit_vector(direction)) {}

    __device__ const point3 &origin() const { return orig; }
    __device__ const vec3 &direction() const { return dir; }

    __device__ point3 at(float t) const { return orig + t * dir; }
};

__device__ inline bool operator==(const ray &r1, const ray &r2)
{
    return r1.origin() == r2.origin() && r1.direction() == r2.direction();
}

__device__ inline bool operator!=(const ray &r1, const ray &r2) { return !(r1 == r2); }

#endif
