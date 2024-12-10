#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

class sphere : public hittable
{
private:
    ray center;
    float radius;
    material *mat;
    aabb bbox;

    __device__ static void get_sphere_uv(const vec3& p, float& u, float& v) {
        auto theta = acosf(-p.y());
        auto phi = atan2f(-p.z(), p.x()) + PI;

        u = phi / (2*PI);
        v = theta / PI;
    }

    __device__ static vec3 random_to_sphere(float radius, float distance_squared, curandState *state) {
        auto r1 = random_float(state);
        auto r2 = random_float(state);
        auto z = 1 + r2*(sqrtf(1-radius*radius/distance_squared) - 1);

        auto phi = 2*PI*r1;
        auto x = cosf(phi) * sqrtf(1-z*z);
        auto y = sinf(phi) * sqrtf(1-z*z);

        return vec3(x, y, z);
    }

public:
    __device__ sphere(const point3 &static_center, float radius, material *mat)
        : center(static_center, vec3(0,0,0)), radius(fmaxf(0.0f, radius)), mat(mat) 
    {
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(static_center - rvec, static_center + rvec);
    }

    __device__ material* get_mat() { return mat; }
    __device__ HittableType get_type() const override { return HittableType::SPHERE; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *state) const override
    {
        vec3 ray_origin_to_sphere_center = center.at(0) - r.origin();
        float a = r.direction().length_squared();
        float h = dot(r.direction(), ray_origin_to_sphere_center);
        float c = ray_origin_to_sphere_center.length_squared() - radius * radius;
        float discriminant = h * h - a * c;

        if (discriminant < 0.0)
        {
            return false;
        }

        float sqrtd = sqrtf(discriminant);
        float root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root))
        {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
            {
                return false;
            }
        }

        rec.t = root;
        rec.hit_point = r.at(rec.t);
        vec3 outward_normal = (rec.hit_point - center.at(0)) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;

        return true;
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ float pdf_value(const point3& origin, const vec3& direction, curandState *state) const override {
        hit_record rec;
        if (!this->hit(ray(origin, direction), interval(0.001, FLT_MAX), rec, state))
            return 0;

        auto dist_squared = (center.at(0) - origin).length_squared();
        auto cos_theta_max = sqrtf(1 - radius*radius/dist_squared);
        auto solid_angle = 2*PI*(1-cos_theta_max);

        return  1 / solid_angle;
    }

    __device__ vec3 random(const point3& origin, curandState *state) const override {
        vec3 direction = center.at(0) - origin;
        auto distance_squared = direction.length_squared();
        onb uvw(direction);
        return uvw.transform(random_to_sphere(radius, distance_squared, state));
    }

};

#endif
