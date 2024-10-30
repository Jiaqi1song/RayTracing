#ifndef COLOR_H
#define COLOR_H

#include "interval.cuh"
#include "vec3.cuh"
#include <stdlib.h>

using color = vec3;


__device__ inline float linear_to_gamma(float linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}


__device__ void write_color(int* pixelData, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Replace NaN components with zero.
    if (r != r) r = 0.0;
    if (g != g) g = 0.0;
    if (b != b) b = 0.0;

    // Apply a linear to gamma transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Translate the [0,1] component values to the byte range [0,255].
    static const interval intensity(0.000, 0.999);
    int rbyte = int(256 * intensity.clamp(r));
    int gbyte = int(256 * intensity.clamp(g));
    int bbyte = int(256 * intensity.clamp(b));

    // Write out the pixel color components.
    pixelData[0] = rbyte;
    pixelData[1] = gbyte;
    pixelData[2] = bbyte;
}

struct Image {

    __device__ Image(int w, int h) {
        width = w;
        height = h;
        data = new int[3 * width * height];
    }

    __device__ void clear(int r, int g, int b) {

        int numPixels = width * height;
        int* ptr = data;
        for (int i=0; i<numPixels; i++) {
            ptr[0] = r;
            ptr[1] = g;
            ptr[2] = b;
            ptr += 3;
        }
    }

    int width;
    int height;
    int* data;
};

__host__ void writePPMImage(const Image* image, const char *filename)
{
    FILE *fp = fopen(filename, "wb");

    if (!fp) {
        fprintf(stderr, "Error: could not open %s for write\n", filename);
        exit(1);
    }

    // write ppm header
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", image->width, image->height);
    fprintf(fp, "255\n");

    for (int j = 0; j < image->height; j++) {
        for (int i = 0; i < image->width; i++) {
            const int* ptr = &image->data[3 * (j*image->width + i)];
            fprintf(fp, "%d %d %d\n", ptr[0], ptr[1], ptr[2]);
        }
    }

    fclose(fp);
}


#endif
