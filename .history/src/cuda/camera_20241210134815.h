#ifndef CAMERA_H
#define CAMERA_H

#include "interval.h"
#include "color.h"
#include "hittable.h"
#include "material.h"

enum CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class camera
{
  private:
    color c1 = color(1.0f, 1.0f, 1.0f);
    color c2 = color(0.5f, 0.7f, 1.0f);

    float pixel_samples_scale;
    point3 camera_center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 u, v, w;
    vec3 defocus_disk_u;
    vec3 defocus_disk_v;
    int sqrt_spp;
    float recip_sqrt_spp;

    __device__ color ray_color(const ray &r, hittable_list **world, hittable_list **lights, curandState *state)
    {
        ray cur_ray = r;
        color final_color = color(0.0f, 0.0f, 0.0f);
        color cur_attenuation = color(1.0f, 1.0f, 1.0f);
        StaticStack<hittable*, 16> stack;
        int depth = 0;
        for (; depth < max_depth; depth++)
        {
            hit_record rec;
            if ((*world)->hit(cur_ray, interval(0.001f, FLT_MAX), rec, state, stack))
            {
                scatter_record srec; 
                color color_from_emission = rec.mat->emitted(rec, rec.u, rec.v, rec.hit_point);
                
                if (rec.mat->scatter(cur_ray, rec, srec, state))
                {
                    if (srec.skip_pdf) {
                        cur_attenuation *= srec.attenuation;
                        cur_ray = srec.skip_pdf_ray;
                        continue;
                    }  
                    
                    vec3 light_pdf = (*lights)->random(rec.hit_point, state);
                    vec3 scatter_direction = srec.generated_pdf;
                    
                    scatter_direction = mixture_pdf_generate(light_pdf, scatter_direction, state);
                    ray scattered = ray(rec.hit_point, scatter_direction);

                    float light_pdf_value = (*lights)->pdf_value(rec.hit_point, scattered.direction(), state);
                    float next_ray_sampling_pdf = srec.pdf_value;
                    next_ray_sampling_pdf = mixture_pdf_value(light_pdf_value, next_ray_sampling_pdf);

                    auto scattering_pdf = rec.mat->scattering_pdf(cur_ray, rec, scattered, state);
                    color attenuation = (srec.attenuation * scattering_pdf / next_ray_sampling_pdf);

                    final_color += cur_attenuation * color_from_emission;
                    cur_attenuation = cur_attenuation * attenuation;
                    cur_ray = scattered;
                    continue;
                }
                else
                {
                    final_color += cur_attenuation * color_from_emission;
                    break;
                }
            }
            else
            {
                final_color += cur_attenuation * background;
                break;
            }
        }
        if (depth >= max_depth) {
            return color(0,0,0);
        }
        return final_color;
    }

    __device__ point3 defocus_disk_sample(curandState *state) const
    {
        vec3 vec = random_in_unit_disk(state);
        return camera_center + (vec[0] * defocus_disk_u) + (vec[1] * defocus_disk_v);
    }

    __device__ ray get_ray(int i, int j, int s_i, int s_j, curandState *state) const 
    {
        auto offset = sample_square_stratified(s_i, s_j, state);
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample(state);
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    __device__ vec3 sample_square_stratified(int s_i, int s_j, curandState *state) const {
        auto px = ((s_i + random_float(state)) * recip_sqrt_spp) - 0.5;
        auto py = ((s_j + random_float(state)) * recip_sqrt_spp) - 0.5;

        return vec3(px, py, 0);
    }

    __device__ color compute_pixel_color(int i, int j, hittable_list **d_world, hittable_list **lights, curandState *state)
    {
        color pixel_color;
        ray r;
        for (int s_j = 0; s_j < sqrt_spp; s_j++) {
            for (int s_i = 0; s_i < sqrt_spp; s_i++) {
                ray r = get_ray(i, j, s_i, s_j, state);
                pixel_color += ray_color(r, d_world, lights, state);
            }
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
    color background;

    vec3 direction;
    float init_theta;
    float init_phi;
    float zoom_scale;
    float step_scale;

    __device__ camera() { initialize(); }

    __device__ camera(int image_width, int image_height)
    {
        this->image_width = max(image_width, 100);
        this->image_height = max(image_height, 100);

        initialize();
    }

    __device__ camera(int image_width, int image_height, int samples_per_pixel, int max_depth, float vfov,
                      point3 lookfrom, point3 lookat, vec3 vup, float defocus_angle, float focus_dist, color background)
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
        this->background = background;

        this->direction = lookfrom - lookat;
        this->init_theta = acosf(this->direction.y() / distance(lookfrom, lookat));
        this->init_phi = atan2f(this->direction.x(), this->direction.z());
        this->zoom_scale = 1.005;
        this->step_scale = 0.6;

        initialize();
    }

    __device__ void initialize()
    {
        sqrt_spp = int(sqrtf(samples_per_pixel));
        pixel_samples_scale = 1.0 / (sqrt_spp * sqrt_spp);
        recip_sqrt_spp = 1.0 / sqrt_spp;

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

    __device__ void render(hittable_list **d_world, hittable_list **lights, int i, int j, curandState *state, uint8_t *output)
    {
        int pixel_index = j * image_width + i;
        color c = compute_pixel_color(i, j, d_world, lights, state);
        translate(c, pixel_index, output);
    }

    __device__ void camera_rotate() {
        float radialDistance = distance(lookfrom, lookat);
        lookfrom = lookat + vec3(
            radialDistance * sinf(init_theta) * sinf(init_phi),
            radialDistance * cosf(init_theta),
            radialDistance * sinf(init_theta) * cosf(init_phi)
        );

        init_phi += 0.1; 
        if (init_phi >= 2 * PI) init_phi -= 2 * PI;

        init_theta -= 0.01;
        if (init_theta <= 0) init_theta = 0;
    }

    __device__ void camera_translate(CameraMovement direction) {
        if (direction == FORWARD) {
            lookfrom += w * step_scale;
            lookat += w * step_scale;
            focus_dist += step_scale;
        }
        if (direction == BACKWARD) {
            lookfrom += w * -step_scale;
            lookat += w * -step_scale;
            focus_dist -= step_scale;
        }
        if (direction == LEFT) {
            lookfrom += u * -step_scale;
            lookat += u * -step_scale;
        }
        if (direction == RIGHT) {
            lookfrom += u * step_scale;
            lookat += u * step_scale;
        }
        if (direction == UP) {
            lookfrom += v * step_scale;
            lookat += v * step_scale;
        }
        if (direction == DOWN) {
            lookfrom += v * -step_scale;
            lookat += v * -step_scale;
        }
    }

};

#endif
