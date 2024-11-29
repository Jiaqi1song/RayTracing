#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

class sphere : public hittable
{
private:
    point3 center;
    float radius;
    material *mat;
    aabb bbox;

public:
    __device__ sphere(const point3 &center, float radius, material *mat)
        : center(center), radius(fmaxf(0.0f, radius)), mat(mat) 
    {
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(center - rvec, center + rvec);
    }

    __device__ material* get_mat() { return mat; }
    __device__ HittableType get_type() const override { return HittableType::SPHERE; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec) const override
    {
        vec3 ray_origin_to_sphere_center = center - r.origin();
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
        rec.hit_point = r.at(root);
        rec.mat = mat;
        vec3 outward_normal = (rec.hit_point - center) / radius;
        rec.set_face_normal(r, outward_normal);
        return true;
    }

    __device__ aabb bounding_box() const override { return bbox; }
};

#endif
