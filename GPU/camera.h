#ifndef CAMERA_H
#define CAMERA_H

#include "interval.h"
#include "aabb.h"
#include "color.h"
#include "hittable.h"
#include "material.h"

class camera
{
  private:
    color c1 = color(1.0f, 1.0f, 1.0f);
    color c2 = color(0.5f, 0.7f, 0.8f);

    float pixel_samples_scale;
    point3 camera_center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 u, v, w;
    vec3 defocus_disk_u;
    vec3 defocus_disk_v;

    __device__ void initialize()
    {
        pixel_samples_scale = 1.0f / samples_per_pixel;

        camera_center = lookfrom;
        float theta = degrees_to_radians(vfov);
        float viewport_height = 2.0f * tanf(theta / 2.0f) * focus_dist;
        float viewport_width = viewport_height * (float(image_width) / image_height);

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        point3 viewport_upper_left = camera_center - focus_dist * w - viewport_u / 2.0f - viewport_v / 2.0f;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        float defocus_radius = focus_dist * tanf(degrees_to_radians(defocus_angle / 2.0f));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ color ray_color(const ray &r, hittable_list **world, curandState *state)
    {
        ray cur_ray = r;
        color cur_attenuation = color(1.0f, 1.0f, 1.0f);
        StaticStack<hittable*, 16> stack;
        for (int i = 0; i < max_depth; i++)
        {
            hit_record rec;
            if ((*world)->hit(cur_ray, interval(0.001f, FLT_MAX), rec, &stack))
            {
                ray scattered;
                color attenuation;
                if (rec.mat->scatter(cur_ray, rec, attenuation, scattered, state))
                {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else
                {
                    return color();
                }
            }
            else
            {
                float a = 0.5f * (r.direction().y() + 1.0f);
                return cur_attenuation * lerp(c1, c2, a);
            }
        }
        return color();
    }

    __device__ point3 defocus_disk_sample(curandState *state) const
    {
        vec3 vec = random_in_unit_disk(state);
        return camera_center + (vec[0] * defocus_disk_u) + (vec[1] * defocus_disk_v);
    }

    __device__ void get_ray(point3 pixel_center, ray &r, curandState *state) const
    {
        point3 pixel_sample = pixel_center +
                              ((random_float(state) - 0.5) * pixel_delta_u) +
                              ((random_float(state) - 0.5) * pixel_delta_v);
        point3 ray_origin = defocus_angle <= 0 ? camera_center : defocus_disk_sample(state);
        r = ray(ray_origin, pixel_sample - ray_origin);
    }

    __device__ color compute_pixel_color(int i, int j, hittable_list **d_world, curandState *state)
    {
        color pixel_color;
        point3 current_pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        ray r;
        for (int sample = 0; sample < samples_per_pixel; sample++)
        {
            get_ray(current_pixel_center, r, state);
            pixel_color += ray_color(r, d_world, state);
        }
        return pixel_samples_scale * pixel_color;
    }

  public:
    int image_width = 1080;
    int image_height = 720;
    int samples_per_pixel = 10;
    int max_depth = 10;

    float vfov = 90.0f;
    point3 lookfrom = point3(0.0f, 0.0f, 0.0f);
    point3 lookat = point3(0.0f, 0.0f, -1.0f);
    vec3 vup = vec3(0.0f, 1.0f, 0.0f);

    float defocus_angle = 0.0f;
    float focus_dist = 10.0f;

    __device__ camera() { initialize(); }

    __device__ camera(int image_width, int image_height)
    {
        this->image_width = max(image_width, 100);
        this->image_height = max(image_height, 100);

        initialize();
    }

    __device__ camera(int image_width, int image_height, int samples_per_pixel, int max_depth, float vfov,
                      point3 lookfrom, point3 lookat, vec3 vup, float defocus_angle, float focus_dist)
    {
        this->image_width = max(image_width, 100);
        this->image_height = max(image_height, 100);
        this->samples_per_pixel = min(max(samples_per_pixel, 4), 500);
        this->max_depth = min(max(max_depth, 5), 50);
        this->vfov = fminf(fmaxf(vfov, 1.0f), 179.0f);
        this->lookfrom = lookfrom;
        this->lookat = lookat;
        this->vup = vup;
        this->defocus_angle = fminf(fmaxf(defocus_angle, 0.0f), 179.0f);
        this->focus_dist = fmaxf(focus_dist, 1.0f);

        initialize();
    }

    __device__ void render(hittable_list **d_world, int i, int j, curandState *state, uint8_t *output)
    {
        int pixel_index = j * image_width + i;
        color c = compute_pixel_color(i, j, d_world, state);
        translate(c, pixel_index, output);
    }
};

#endif
