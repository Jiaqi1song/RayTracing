#ifndef TEXTURE_H
#define TEXTURE_H

#include "perlin.h"
#include "color.h"

class texture_custum {
  public:
    __device__ virtual ~texture_custum() {};
    __device__ virtual color value(float u, float v, const point3& p) const = 0;
};


class solid_color : public texture_custum {
  public:
    __device__ solid_color(const color& albedo) : albedo(albedo) {}
    __device__ solid_color(float red, float green, float blue) : solid_color(color(red,green,blue)) {}
    __device__ color value(float u, float v, const point3& p) const override {return albedo; }

  private:
    color albedo;
};


class checker_texture : public texture_custum {
  public:
    __device__ checker_texture(float scale, texture_custum *even, texture_custum *odd)
      : inv_scale(1.0 / scale), even(even), odd(odd) {}

    __device__ checker_texture(float scale, const color& c1, const color& c2)
      : checker_texture(scale, new solid_color(c1), new solid_color(c2)) {}

    __device__ color value(float u, float v, const point3& p) const override 
    {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

  private:
    float inv_scale;
    texture_custum *even;
    texture_custum *odd;
};


class noise_texture : public texture_custum {
  public:
    __device__ noise_texture(float scale, curandState *state) : scale(scale), noise(state) {}

    __device__ color value(float u, float v, const point3& p) const override {
        return color(.5, .5, .5) * (1 + sinf(scale * p.z() + 10 * noise.turb(p, 7)));
    }

  private:
    perlin noise;
    float scale;
};

class image_texture : public texture_custum {
  public:
    __device__ image_texture(unsigned char* data, int image_width, int image_height) : image_data(data), image_width(image_width), image_height(image_height) {}

    __device__ color value(float u, float v, const point3& p) const override {

        if (image_height <= 0) return color(0,1,1);
        u = interval(0,1).clamp(u);
        v = 1.0 - interval(0,1).clamp(v);  

        auto i = int(u * image_width);
        auto j = int(v * image_height);

        i = fmaxf(0, fminf(i, image_width - 1));
        j = fmaxf(0, fminf(j, image_height - 1));

        int bytes_per_scanline = image_width * 3;
        auto pixel = image_data + j * bytes_per_scanline + i * 3;
        auto color_scale = 1.0 / 255.0;
        return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
    }

  private:
    unsigned char *image_data;
    int image_width;
    int image_height;
};


#endif