#ifndef COLOR_H
#define COLOR_H

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#pragma nv_diag_suppress 550
#include "stb_image.h"
#pragma nv_diag_default 550

#include "vec.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

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
inline void write_color(uint8_t *h_output, int image_width, int image_height, const char *filename)
{

    FILE *fp = fopen(filename, "wb");

    if (!fp) {
        fprintf(stderr, "Error: could not open %s for write\n", filename);
        exit(1);
    }

    // write ppm header
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", image_width, image_height);
    fprintf(fp, "255\n");

    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int start_write_index = 3 * (i * image_width + j);
            fprintf(fp, "%d %d %d\n", static_cast<int>(h_output[start_write_index]), 
                                      static_cast<int>(h_output[start_write_index+1]), 
                                      static_cast<int>(h_output[start_write_index+2]));
        }
    }

    fclose(fp);
}


// Host-only function for parsing object
inline void parse_obj(const char* filename, 
            float* vertices, 
            float* indices, 
            int& nPoints, 
            int& nTriangles) {

    std::ifstream objFile(filename); 
    if (!objFile.is_open()) {
        std::cerr << "Error: Cannot open the OBJ file: " << filename << std::endl;
        return;
    }

    std::vector<float> points;   
    std::vector<float> idxVertex;  

    std::string line;
    nPoints = 0;
    nTriangles = 0;

    while (std::getline(objFile, line)) {
        std::stringstream ss(line);
        std::string label;
        ss >> label;

        if (label == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            points.push_back(x);
            points.push_back(y);
            points.push_back(z);
            nPoints++;
        } else if (label == "f") {
            int v1, v2, v3;
            ss >> v1 >> v2 >> v3;
            idxVertex.push_back(v1 - 1); // Convert to 0-based indexing
            idxVertex.push_back(v2 - 1);
            idxVertex.push_back(v3 - 1);
            nTriangles++;
        }
    }

    objFile.close(); 

    std::memcpy(vertices, points.data(), points.size() * sizeof(float));
    std::memcpy(indices, idxVertex.data(), idxVertex.size() * sizeof(float));

    // Perform centering and scaling
    float meanX = 0, meanY = 0, meanZ = 0;
    for (int i = 0; i < nPoints; ++i) {
        meanX += vertices[i * 3 + 0];
        meanY += vertices[i * 3 + 1];
        meanZ += vertices[i * 3 + 2];
    }
    meanX /= nPoints;
    meanY /= nPoints;
    meanZ /= nPoints;

    for (int i = 0; i < nPoints; ++i) {
        vertices[i * 3 + 0] -= meanX;
        vertices[i * 3 + 1] -= meanY;
        vertices[i * 3 + 2] -= meanZ;
    }

    float maxDistance = 0.0f;
    for (int i = 0; i < nPoints; ++i) {
        float dist = sqrt(vertices[i * 3 + 0] * vertices[i * 3 + 0] +
                          vertices[i * 3 + 1] * vertices[i * 3 + 1] +
                          vertices[i * 3 + 2] * vertices[i * 3 + 2]);
        maxDistance = std::max(maxDistance, dist);
    }

    for (int i = 0; i < nPoints; ++i) {
        vertices[i * 3 + 0] /= maxDistance;
        vertices[i * 3 + 1] /= maxDistance;
        vertices[i * 3 + 2] /= maxDistance;
    }
}

// Host-only function for loading images
inline unsigned char *load_image(const char* image_filename, int& width, int& height) {

    auto filename = std::string(image_filename);
    auto imagedir = getenv("RTW_IMAGES");

    const int      bytes_per_pixel = 3;
    float         *fdata = nullptr;        
    unsigned char *bdata = nullptr;         
    int            image_width = 0;        
    int            image_height = 0;       

    auto n = bytes_per_pixel; 
    fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);

    if (fdata == nullptr)  {
        std::cerr << "Error: Cannot Read Image File: " << filename << std::endl;
        return nullptr;
    }
    width = image_width;
    height = image_height;

    int total_bytes = image_width * image_height * bytes_per_pixel;
    bdata = new unsigned char[total_bytes];

    auto *bptr = bdata;
    auto *fptr = fdata;

    float value;
    for (auto i=0; i < total_bytes; i++, fptr++, bptr++) {
        value = *fptr;
        if (value <= 0.0) {
            *bptr = 0;
        } else if (1.0 <= value) {
            *bptr = 255;
        } else {
            *bptr = static_cast<unsigned char>(256.0 * value);
        }
    }

    return bdata;
}

#endif
