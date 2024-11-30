#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "interval.h"
#include "aabb.h"

class material;
enum HittableType { SPHERE, QUAD, MEDIUM, HITTABLE_LIST, BVH};

class hit_record
{
public:
    float t = 0.0f;
    float u;
    float v;
    point3 hit_point = point3();
    vec3 normal_vector = vec3();
    bool front_face = false;
    material *mat;

    __device__ inline void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0.0f;
        normal_vector = front_face ? outward_normal : -outward_normal;
    }
};

class hittable
{
public:
    __device__ virtual ~hittable() {};
    __device__ virtual bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *state) const = 0;
    __device__ virtual aabb bounding_box() const = 0;
    __device__ virtual HittableType get_type() const = 0; 
};

class hittable_list : public hittable
{
public:
    hittable **objects;
    int obj_num;

    __device__ hittable_list(hittable **objects, int obj_num) : objects(objects), obj_num(obj_num) 
    {
        for (int i = 0; i < obj_num; ++i)
        {
            bbox = aabb(bbox, objects[i]->bounding_box());
        }
    }

    __device__ HittableType get_type() const override { return HittableType::HITTABLE_LIST; }

    __device__ bool hit(const ray &r, const interval &ray_t, hit_record &rec, curandState *state) const override
    {
        hit_record tmp_rec;
        bool hit_anything = false;
        float closest_t = ray_t.max;

        for (int i = 0; i < obj_num; ++i)
        {
            if (objects[i]->hit(r, interval(ray_t.min, closest_t), tmp_rec, state))
            {
                hit_anything = true;
                closest_t = tmp_rec.t;
                rec = tmp_rec;
            }
        }

        return hit_anything;
    }

    __device__ aabb bounding_box() const override { return bbox; }

private:
    aabb bbox;
};



#endif