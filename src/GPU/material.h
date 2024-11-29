#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "hittable.h"
#include "texture.h"

class material
{
public:
    __device__ virtual ~material() {};
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &r_out, curandState *state) const = 0;
    __device__ virtual color emitted(double u, double v, const point3& p) const {return color(0,0,0); }
};

class lambertian : public material
{
private:
    texture_custum *tex;

public:
    __device__ lambertian(const color& albedo) : tex(new solid_color(albedo)) {}
    __device__ lambertian(texture_custum *tex) : tex(tex) {}

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &r_out, curandState *state) const override
    {
        vec3 scatter_direction = rec.normal_vector + random_unit_vector(state);
        while (scatter_direction.near_zero())
            scatter_direction = rec.normal_vector + random_unit_vector(state);
        r_out = ray(rec.hit_point, scatter_direction);
        attenuation = tex->value(rec.u, rec.v, rec.hit_point);
        return true;
    }
};

class metal : public material
{
private:
    color albedo;
    float fuzz;

public:
    __device__ metal(const color &albedo, float fuzz) : albedo(albedo), fuzz(fminf(fmaxf(fuzz, 0.0f), 1.0f)) {}

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &r_out, curandState *state) const override
    {
        vec3 reflect_direction = reflect(r_in.direction(), rec.normal_vector);
        reflect_direction = unit_vector(reflect_direction) + fuzz * random_unit_vector(state);
        r_out = ray(rec.hit_point, reflect_direction);
        attenuation = albedo;
        return (dot(reflect_direction, rec.normal_vector) > 0.0f);
    }
};

class dielectric : public material
{
private:
    float refraction_index;
    color dielectrics_attenuation = color(1.0f, 1.0f, 1.0f);

    __device__ static float reflectance(float cosine, float refraction_index)
    {
        float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * powf((1.0f - cosine), 5);
    }

public:
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &r_out, curandState *state) const override
    {
        attenuation = dielectrics_attenuation;
        float ri = rec.front_face ? (1.0f / refraction_index) : refraction_index;

        float cos_theta = fminf(dot(-r_in.direction(), rec.normal_vector), 1.0f);
        float sin_theta = sqrtf(1.0 - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_float(state))
            direction = reflect(r_in.direction(), rec.normal_vector);
        else
            direction = refract(r_in.direction(), rec.normal_vector, ri);

        r_out = ray(rec.hit_point, direction);
        return true;
    }
};

class diffuse_light : public material 
{
  public:
    __device__ diffuse_light(texture_custum *tex) : tex(tex) {}
    __device__ diffuse_light(const color& emit) : tex(new solid_color(emit)) {}

    __device__ color emitted(double u, double v, const point3& p) const override {
        return tex->value(u, v, p);
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &r_out, curandState *state) const override {
        return false;
    }

  private:
    texture_custum *tex;
};


#endif
