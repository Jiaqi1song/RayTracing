#ifndef QUAD_H
#define QUAD_H

#include "hittable.h"
#include <cuda_runtime.h>

class quad : public hittable {
  public:
    __device__ quad(const point3& Q, const vec3& u, const vec3& v, material *mat)
      : Q(Q), u(u), v(v), mat(mat)
    {
        auto n = cross(u, v);
        normal = unit_vector(n);
        
        vec3 Q_vec3 = vec3(Q.x(), Q.y(), Q.z());
        D = dot(normal, Q_vec3);
        w = n / dot(n,n);

        set_bounding_box();
    }

    __device__ material* get_mat() { return mat; }
    __device__ HittableType get_type() const override { return HittableType::QUAD; }
    
    __device__ virtual void set_bounding_box() {
        auto bbox_diagonal1 = aabb(Q, Q + u + v);
        auto bbox_diagonal2 = aabb(Q + u, Q + v);
        bbox = aabb(bbox_diagonal1, bbox_diagonal2);
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ bool hit(const ray& r, const interval& ray_t, hit_record& rec, curandState *state) const override {
        auto denom = dot(normal, r.direction());

        // No hit if the ray is parallel to the plane.
        if (fabs(denom) < 1e-8)
            return false;

        // Return false if the hit point parameter t is outside the ray interval.
        vec3 r_vec3 = vec3(r.origin().x(), r.origin().y(), r.origin().z());
        auto t = (D - dot(normal, r_vec3)) / denom;
        if (!ray_t.contains(t))
            return false;

        // Determine if the hit point lies within the planar shape using its plane coordinates.
        auto intersection = r.at(t);
        vec3 planar_hitpt_vector = intersection - Q;
        auto alpha = dot(w, cross(planar_hitpt_vector, v));
        auto beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec))
            return false;

        // Ray hits the 2D shape; set the rest of the hit record and return true.
        rec.t = t;
        rec.hit_point = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, normal);

        return true;
    }

    __device__ virtual bool is_interior(float a, float b, hit_record& rec) const {
        interval unit_interval = interval(0, 1);
        // Given the hit point in plane coordinates, return false if it is outside the
        // primitive, otherwise set the hit record UV coordinates and return true.

        if (!unit_interval.contains(a) || !unit_interval.contains(b))
            return false;

        rec.u = a;
        rec.v = b;
        return true;
    }

  private:
    point3 Q;
    vec3 u, v;
    vec3 w;
    material *mat;
    aabb bbox;
    vec3 normal;
    float D;
};


#endif