#ifndef INTERVAL_H
#define INTERVAL_H

#include "vec.h"

class interval
{
  public:
    float min, max;

    __device__ interval() : min(+FLT_MAX), max(-FLT_MAX) {}
    __device__ interval(float min, float max) : min(min), max(max) {}
    __device__ interval(const interval& a, const interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    __device__ float size() const
    {
        if (max >= min - EPSILON)
            return max - min;
        return 0.0f;
    }

    __device__ bool contains(float x) const { return min <= x && x <= max; }

    __device__ bool surrounds(float x) const { return min < x && x < max; }

    __device__ float clamp(float f) const { return fminf(fmaxf(f, min), max); }

    __device__ interval expand(float delta) const 
    {
        auto padding = delta/2;
        return interval(min - padding, max + padding);
    }

    __device__ static interval empty() { return interval(+FLT_MAX, -FLT_MAX); }

    __device__ static interval universe() { return interval(-FLT_MAX, +FLT_MAX); }

};

__device__ interval operator+(const interval& ival, float displacement) {return interval(ival.min + displacement, ival.max + displacement); }

__device__ interval operator+(float displacement, const interval& ival) {return ival + displacement; }

#endif
