#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "hittable.h"
#include "material.h"
#include "texture.h"


class constant_medium : public hittable {
  public:
    __device__ constant_medium(hittable *boundary, float density, texture_custum *tex)
      : boundary(boundary), neg_inv_density(-1/density),
        phase_function(new isotropic(tex))
    {}

    __device__ constant_medium(hittable *boundary, float density, const color& albedo)
      : boundary(boundary), neg_inv_density(-1/density),
        phase_function(new isotropic(albedo))
    {}

    __device__ material* get_mat() { return phase_function; }
    __device__ HittableType get_type() const override { return HittableType::MEDIUM; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *state, StaticStack<hittable*, 16> *stack=nullptr) const override {
        hit_record rec1, rec2;

        if (!boundary->hit(r, interval::universe(), rec1, state, stack))
            return false;

        if (!boundary->hit(r, interval(rec1.t+0.0001, FLT_MAX), rec2, state, stack))
            return false;

        if (rec1.t < ray_t.min) rec1.t = ray_t.min;
        if (rec2.t > ray_t.max) rec2.t = ray_t.max;

        if (rec1.t >= rec2.t)
            return false;

        if (rec1.t < 0)
            rec1.t = 0;

        auto ray_length = r.direction().length();
        auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
        auto hit_distance = neg_inv_density * logf(random_float(state));

        if (hit_distance > distance_inside_boundary)
            return false;

        rec.t = rec1.t + hit_distance / ray_length;
        rec.hit_point = r.at(rec.t);

        rec.normal_vector = vec3(1,0,0); 
        rec.front_face = true;     
        rec.mat = phase_function;

        return true;
    }

    __device__ aabb bounding_box() const override { return boundary->bounding_box(); }

  private:
    hittable *boundary;
    float neg_inv_density;
    material *phase_function;
};


#endif