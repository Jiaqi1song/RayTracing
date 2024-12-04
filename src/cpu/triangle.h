#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "stb_image_utils.h"
#include "bvh.h"

#include <vector>
#include <array>

class triangle : public hittable {
public:
    triangle(const vec3 vs[3], shared_ptr<material> mat, bool cull = false)
        : EPSILON(0.000001), mat(mat), backCulling(cull) {
        for (int i = 0; i < 3; i++) {
            vertices[i] = vs[i];
        }
        compute_normal();
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3 edge1 = vertices[1] - vertices[0];
        vec3 edge2 = vertices[2] - vertices[0];
        vec3 h = cross(r.direction(), edge2);
        float a = dot(edge1, h);

        if (backCulling && a < EPSILON)
            return false;

        if (std::fabs(a) < EPSILON)
            return false;

        float f = 1.0 / a;
        vec3 s = r.origin() - vertices[0];
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
        rec.p = r.at(t);
        rec.normal = normal;
        rec.mat = mat;
        rec.set_face_normal(r, rec.normal);
        rec.u = u;
        rec.v = v;

        return true;
    }

    aabb bounding_box() const override {
        float minX = std::min(vertices[0].x(), std::min(vertices[1].x(), vertices[2].x()));
        float minY = std::min(vertices[0].y(), std::min(vertices[1].y(), vertices[2].y()));
        float minZ = std::min(vertices[0].z(), std::min(vertices[1].z(), vertices[2].z()));

        float maxX = std::max(vertices[0].x(), std::max(vertices[1].x(), vertices[2].x()));
        float maxY = std::max(vertices[0].y(), std::max(vertices[1].y(), vertices[2].y()));
        float maxZ = std::max(vertices[0].z(), std::max(vertices[1].z(), vertices[2].z()));

        return aabb(point3(minX, minY, minZ), point3(maxX, maxY, maxZ));
    }

private:
    vec3 vertices[3];
    vec3 normal;
    shared_ptr<material> mat;
    bool backCulling;
    const float EPSILON;

    void compute_normal() {
        vec3 edge1 = vertices[1] - vertices[0];
        vec3 edge2 = vertices[2] - vertices[0];
        normal = unit_vector(cross(edge1, edge2));
    }
};

inline hittable_list build_mesh(const char* filename, shared_ptr<material> mat, double scale, bool use_bvh) {

    hittable_list mesh_obj;
    std::vector<vec3> vertices(5000); 
    std::vector<vec3> indices(5000);

    // Parse the OBJ file
    int nPoints = 0, nTriangles = 0;
    parse_obj(filename, vertices, indices, nPoints, nTriangles);

    // Scaling
    for(vec3& p : vertices) p *= scale;

    // Create triangles and add them to the mesh
    for (int i = 0; i < nTriangles; i++) {
        vec3 v0 = vertices[static_cast<int>(indices[i].x())];
        vec3 v1 = vertices[static_cast<int>(indices[i].y())];
        vec3 v2 = vertices[static_cast<int>(indices[i].z())];

        auto obj = make_shared<triangle>(std::array<vec3, 3>{v0, v1, v2}.data(), mat, false);
        mesh_obj.add(obj);
    }

    if (use_bvh) mesh_obj = hittable_list(make_shared<bvh_node>(mesh_obj));

    return mesh_obj;
}

#endif 