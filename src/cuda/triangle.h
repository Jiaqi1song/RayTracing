#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hittable.h"
#include "material.h"
#include "vec.h"

class triangle : public hittable {
public:
    __device__ triangle(const vec3 v1, const vec3 v2, const vec3 v3, material *mat, bool cull = false)
        : EPSILON(0.000001), mat(mat), backCulling(cull) {
        vertices1 = v1;
        vertices2 = v2;
        vertices3 = v3;
        compute_normal();
    }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *state) const override {
        vec3 edge1 = vertices2 - vertices1;
        vec3 edge2 = vertices3 - vertices1;
        vec3 h = cross(r.direction(), edge2);
        float a = dot(edge1, h);

        if (backCulling && a < EPSILON)
            return false;

        if (fabs(a) < EPSILON)
            return false;

        float f = 1.0 / a;

        vec3 r_origin_vec = vec3(r.origin().x(), r.origin().y(), r.origin().z());
        vec3 s = r_origin_vec - vertices1;
        float u = f * dot(s, h);

        if (u < 0.0 || u > 1.0)
            return false;

        vec3 q = cross(s, edge1);
        float v = f * dot(r.direction(), q);

        if (v < 0.0 || u + v > 1.0)
            return false;

        float t = f * dot(edge2, q);
        if (!ray_t.contains(t))
            return false;

        rec.t = t;
        rec.hit_point = r.at(t);
        rec.normal_vector = normal;
        rec.mat = mat;
        rec.set_face_normal(r, rec.normal_vector);
        rec.u = u;
        rec.v = v;

        return true;
    }

    __device__ HittableType get_type() const override { return HittableType::TRIANGLE; }
    __device__ material* get_mat() { return mat; }

    __device__ aabb bounding_box() const override {
        float minX = fminf(vertices1.x(), fminf(vertices2.x(), vertices3.x()));
        float minY = fminf(vertices1.y(), fminf(vertices2.y(), vertices3.y()));
        float minZ = fminf(vertices1.z(), fminf(vertices2.z(), vertices3.z()));

        float maxX = fmaxf(vertices1.x(), fmaxf(vertices2.x(), vertices3.x()));
        float maxY = fmaxf(vertices1.y(), fmaxf(vertices2.y(), vertices3.y()));
        float maxZ = fmaxf(vertices1.z(), fmaxf(vertices2.z(), vertices3.z()));

        return aabb(point3(minX, minY, minZ), point3(maxX, maxY, maxZ));
    }

private:
    vec3 vertices1;
    vec3 vertices2;
    vec3 vertices3;
    vec3 normal;
    material *mat;
    bool backCulling;
    const float EPSILON;

    __device__ void compute_normal() {
        vec3 edge1 = vertices2 - vertices1;
        vec3 edge2 = vertices3 - vertices1;
        normal = unit_vector(cross(edge1, edge2));
    }
};


__device__ inline vec3 transform_mesh(const vec3& input_vec, const vec3& translate_vec, float angle) {
    float radians = degrees_to_radians(angle);
    float sin_theta = sinf(radians);
    float cos_theta = cosf(radians);

    float x = input_vec.x();
    float y = input_vec.y();
    float z = input_vec.z();

    float rotated_x = cos_theta * x + sin_theta * z;
    float rotated_y = y;  
    float rotated_z = -sin_theta * x + cos_theta * z;

    float x1 = rotated_x + translate_vec.x();
    float y1 = rotated_y + translate_vec.y();
    float z1 = rotated_z + translate_vec.z();

    return vec3(x1, y1, z1);
}



#endif 