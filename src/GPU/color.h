#ifndef COLOR_H
#define COLOR_H

#include "vec.h"
#include <cstdint>

class color
{
  protected:
    float _r, _g, _b;

  public:
    __device__ color() : _r(0.0f), _g(0.0f), _b(0.0f) {}
    __device__ color(float r, float g, float b) : _r(r), _g(g), _b(b) {}

    __device__ float r() const { return _r; }
    __device__ float g() const { return _g; }
    __device__ float b() const { return _b; }

    __device__ color &operator+=(const color &v)
    {
        _r += v.r();
        _g += v.g();
        _b += v.b();
        return *this;
    }

    __device__ color &operator-=(const color &v)
    {
        _r -= v.r();
        _g -= v.g();
        _b -= v.b();
        return *this;
    }

    __device__ color &operator*=(const color &v)
    {
        _r *= v.r();
        _g *= v.g();
        _b *= v.b();
        return *this;
    }

    __device__ color &operator*=(float t)
    {
        _r *= t;
        _g *= t;
        _b *= t;
        return *this;
    }

    __device__ color &operator/=(float t)
    {
        assert(t != 0.0f);
        float inv_t = 1.0f / t;
        _r *= inv_t;
        _g *= inv_t;
        _b *= inv_t;
        return *this;
    }

    __device__ static color random(curandState *state)
    {
        return color(random_float(state), random_float(state), random_float(state));
    }

    __device__ static color random(curandState *state, float min, float max)
    {
        return color(random_float(state, min, max), random_float(state, min, max), random_float(state, min, max));
    }

    __device__ color clamp_and_gamma_correct(float gamma = 2.2f)
    {
        float _r_clamped = fminf(fmaxf(_r, 0.0f), 1.0f);
        float _g_clamped = fminf(fmaxf(_g, 0.0f), 1.0f);
        float _b_clamped = fminf(fmaxf(_b, 0.0f), 1.0f);

        return color(powf(_r_clamped, 1.0f / gamma), powf(_g_clamped, 1.0f / gamma), powf(_b_clamped, 1.0f / gamma));
    }
};

__device__ inline bool operator==(const color &u, const color &v)
{
    return fabsf(u.r() - v.r()) < EPSILON && fabsf(u.g() - v.g()) < EPSILON && fabsf(u.b() - v.b()) < EPSILON;
}

__device__ inline bool operator!=(const color &u, const color &v) { return !(u == v); }

__device__ inline color operator+(const color &u, const color &v)
{
    return color(u.r() + v.r(), u.g() + v.g(), u.b() + v.b());
}

__device__ inline color operator-(const color &u, const color &v)
{
    return color(u.r() - v.r(), u.g() - v.g(), u.b() - v.b());
}

__device__ inline color operator*(const color &u, const color &v)
{
    return color(u.r() * v.r(), u.g() * v.g(), u.b() * v.b());
}

__device__ inline color operator*(const color &u, float t) { return color(u.r() * t, u.g() * t, u.b() * t); }

__device__ inline color operator*(float t, const color &v) { return color(t * v.r(), t * v.g(), t * v.b()); }

__device__ inline color operator/(const color &u, float t)
{
    // assert(t != 0.0f);
    return color(u.r() / t, u.g() / t, u.b() / t);
}

__device__ inline color lerp(const color &c1, const color &c2, float t)
{
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return (1.0f - t) * c1 + t * c2;
}

// Translate the color from [0, 1) to 8-bit int [0, 255) for output.
__device__ void translate(color c, int pixel_index, uint8_t *output)
{
    color c_translated = c.clamp_and_gamma_correct();

    int write_index = 3 * pixel_index;
    output[write_index] = static_cast<uint8_t>(255.999f * c_translated.r());
    output[write_index + 1] = static_cast<uint8_t>(255.999f * c_translated.g());
    output[write_index + 2] = static_cast<uint8_t>(255.999f * c_translated.b());
}

// Host-only function for writing color to output stream
inline void write_color(std::ostream &out, uint8_t ir, uint8_t ig, uint8_t ib)
{
    std::cout << static_cast<int>(ir) << " " << static_cast<int>(ig) << " " << static_cast<int>(ib) << "\n";
}

#endif
