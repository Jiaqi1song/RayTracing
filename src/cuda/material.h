#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "hittable.h"
#include "texture.h"
#include "pdf.h"

class scatter_record {
public:
    color attenuation;
    bool skip_pdf;
    pdf *next_pdf;
    ray skip_pdf_ray;
};

class material
{
public:
    __device__ virtual ~material() {};
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, scatter_record& srec, curandState *state) const {return false; };
    __device__ virtual color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p) const {return color(0,0,0); }
    __device__ virtual float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered, curandState *state) const {return 0; }
};

class lambertian : public material
{
private:
    texture_custum *tex;

public:
    __device__ lambertian(const color& albedo) : tex(new solid_color(albedo)) {}
    __device__ lambertian(texture_custum *tex) : tex(tex) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec, curandState *state) const override 
    {
        srec.attenuation = tex->value(rec.u, rec.v, rec.hit_point);
        srec.next_pdf = new cosine_pdf(rec.normal_vector);
        srec.skip_pdf = false;
        return true;
    }

    __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered, curandState *state) const override 
    {
        auto cos_theta = dot(rec.normal_vector, unit_vector(scattered.direction()));
        return cos_theta < 0 ? 0 : cos_theta/PI;
    }
};

class metal : public material
{
private:
    color albedo;
    float fuzz;

public:
    __device__ metal(const color &albedo, float fuzz) : albedo(albedo), fuzz(fminf(fmaxf(fuzz, 0.0f), 1.0f)) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec, curandState *state) const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal_vector);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(state));

        srec.attenuation = albedo;
        srec.skip_pdf = true;
        srec.next_pdf = nullptr;
        srec.skip_pdf_ray = ray(rec.hit_point, reflected);

        return true;
    }

};

class dielectric : public material
{
private:
    float refraction_index;

    __device__ static float reflectance(float cosine, float refraction_index)
    {
        float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * powf((1.0f - cosine), 5);
    }

public:
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec, curandState *state) const override {
        srec.attenuation = color(1.0, 1.0, 1.0);
        srec.skip_pdf = true;
        srec.next_pdf = nullptr;
        float ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fminf(dot(-unit_direction, rec.normal_vector), 1.0);
        float sin_theta = sqrtf(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_float(state))
            direction = reflect(unit_direction, rec.normal_vector);
        else
            direction = refract(unit_direction, rec.normal_vector, ri);

        srec.skip_pdf_ray = ray(rec.hit_point, direction);
        return true;
    }
};

class diffuse_light : public material 
{
  public:
    __device__ diffuse_light(texture_custum *tex) : tex(tex) {}
    __device__ diffuse_light(const color& emit) : tex(new solid_color(emit)) {}

    __device__ color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p) const override 
    {
        if (!rec.front_face)
            return color(0,0,0);
        return tex->value(u, v, p);
    }

  private:
    texture_custum *tex;
};

class isotropic : public material {
  public:
    __device__ isotropic(const color& albedo) : tex(new solid_color(albedo)) {}
    __device__ isotropic(texture_custum *tex) : tex(tex) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec, curandState *state) const override 
    {
        srec.attenuation = tex->value(rec.u, rec.v, rec.hit_point);
        srec.next_pdf = new sphere_pdf();
        srec.skip_pdf = false;
        return true;
    }

    __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered, curandState *state) const override {
        return 1 / (4 * PI);
    }

  private:
    texture_custum *tex;
};

#endif
