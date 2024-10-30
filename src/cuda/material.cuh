#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.cuh"
#include "pdf.cuh"
#include "texture.cuh"


class scatter_record {
  public:
    color attenuation;
    shared_ptr<pdf> pdf_ptr;
    bool skip_pdf;
    ray skip_pdf_ray;
};


class material {
  public:
    __device__ virtual ~material() = default;

    __device__ virtual color emitted(
        const ray& r_in, const hit_record& rec, float u, float v, const point3& p
    ) const {
        return color(0,0,0);
    }

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const {
        return false;
    }

    __device__ virtual float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    const {
        return 0;
    }
};


class lambertian : public material {
  public:
    __device__ lambertian(const color& albedo) : tex(new solid_color(albedo)) {}
    __device__ lambertian(shared_ptr<texture> tex) : tex(tex) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        srec.attenuation = tex->value(rec.u, rec.v, rec.p);
        srec.pdf_ptr = new cosine_pdf(rec.normal);
        srec.skip_pdf = false;
        return true;
    }

    __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    __device__ const override {
        auto cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
        return cos_theta < 0 ? 0 : cos_theta/pi;
    }

  private:
    shared_ptr<texture> tex;
};


class metal : public material {
  public:
    __device__ metal(const color& albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector());

        srec.attenuation = albedo;
        srec.pdf_ptr = nullptr;
        srec.skip_pdf = true;
        srec.skip_pdf_ray = ray(rec.p, reflected, r_in.time());

        return true;
    }

  private:
    color albedo;
    float fuzz;
};


class dielectric : public material {
  public:
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        srec.attenuation = color(1.0, 1.0, 1.0);
        srec.pdf_ptr = nullptr;
        srec.skip_pdf = true;
        float ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_float())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        srec.skip_pdf_ray = ray(rec.p, direction, r_in.time());
        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float refraction_index;

    __device__ static float reflectance(float cosine, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
};


class diffuse_light : public material {
  public:
    __device__ diffuse_light(shared_ptr<texture> tex) : tex(tex) {}
    __device__ diffuse_light(const color& emit) : tex(new solid_color(emit)) {}

    __device__ color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p)
    __device__ const override {
        if (!rec.front_face)
            return color(0,0,0);
        return tex->value(u, v, p);
    }

  private:
    shared_ptr<texture> tex;
};


class isotropic : public material {
  public:
    __device__ isotropic(const color& albedo) : tex(new solid_color(albedo)) {}
    __device__ isotropic(shared_ptr<texture> tex) : tex(tex) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, scatter_record& srec) const override {
        srec.attenuation = tex->value(rec.u, rec.v, rec.p);
        srec.pdf_ptr = new sphere_pdf();
        srec.skip_pdf = false;
        return true;
    }

    __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered)
    __device__ const override {
        return 1 / (4 * pi);
    }

  private:
    shared_ptr<texture> tex;
};


#endif
